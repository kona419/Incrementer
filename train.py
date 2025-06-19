
import numpy as np
import torch
import argparse
import argparsers
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
from torch.utils import data


# from segm.utils import distributed
from torch import distributed
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
# import segm.utils.torch as ptu
import config
from utils.logger import Logger

from model.factory import create_segmenter
from optim.factory import create_optimizer, create_scheduler
from optim.scheduler import PolyLR
from model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress
from timm.optim import create_optimizer


from dataset import VOCSegmentationIncremental, AdeSegmentationIncremental
from dataset import transform
import tasks
from engine import Trainer
from stream_metrics import StreamSegMetrics

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"

# from argparser import get_argparser

def save_ckpt(path, model, model_without_ddp, optimizer, scheduler, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        # "trainer_state": trainer.state_dict(),
        "n_cls": model_without_ddp.n_cls
    }
    torch.save(state, path)

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = transform.Compose([
            transform.Resize(size=opts.crop_size),
            transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels
    # print(labels)
    # print(labels_old)

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(root=opts.data_root, train=True, transform=train_transform,
                        labels=list(labels), labels_old=list(labels_old),
                        idxs_path=path_base + f"/train-{opts.step}.npy",
                        masking=not opts.no_mask, overlap=opts.overlap)

    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst)-train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(root=opts.data_root, train=False, transform=val_transform,
                          labels=list(labels), labels_old=list(labels_old),
                          idxs_path=path_base + f"/val-{opts.step}.npy",
                          masking=not opts.no_mask, overlap=True)

    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = dataset(root=opts.data_root, train=opts.val_on_trainset, transform=val_transform,
                       labels=list(labels_cum),
                       idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy")

    return train_dst, val_dst, test_dst, len(labels_cum), labels, labels_old, labels_cum

def expand_class_embeddings(checkpoint, classes, gpu):

    cls_emb_key = next(k for k in checkpoint['model_state'].keys() if 'cls_emb' in k)
    old_cls_emb = checkpoint['model_state'][cls_emb_key]

    new_cls_emb_shape = (old_cls_emb.size(0), sum(classes), old_cls_emb.size(2))
    new_cls_emb = torch.randn(new_cls_emb_shape).to(gpu)

    new_cls_emb[:, :old_cls_emb.size(1), :] = old_cls_emb

    mask_norm_weight_key = next(k for k in checkpoint['model_state'].keys() if 'mask_norm.weight' in k)
    mask_norm_bias_key = next(k for k in checkpoint['model_state'].keys() if 'mask_norm.bias' in k)

    new_mask_norm_weight = torch.randn(sum(classes)).to(gpu)
    new_mask_norm_bias = torch.randn(sum(classes)).to(gpu)

    old_mask_norm_weight = checkpoint['model_state'][mask_norm_weight_key]
    old_mask_norm_bias = checkpoint['model_state'][mask_norm_bias_key]

    new_mask_norm_weight[:old_mask_norm_weight.size(0)] = old_mask_norm_weight
    new_mask_norm_bias[:old_mask_norm_bias.size(0)] = old_mask_norm_bias

    checkpoint['model_state'][cls_emb_key] = new_cls_emb
    checkpoint['model_state'][mask_norm_weight_key] = new_mask_norm_weight
    checkpoint['model_state'][mask_norm_bias_key] = new_mask_norm_bias
    return checkpoint


def main(gpu, opts, ngpus_per_node, world_size):
    
    task_name = f"{opts.task}-{opts.dataset}"
    set = 'overlap'
    paths = f"/scratch/kona419/Incrementer/checkpoints"
    logdir_full = f"/scratch/kona419/Incrementer/log/{task_name}/{set}"
    
    # start distributed mode    
    torch.cuda.set_device(gpu)
    rank = 0
    
    rank = rank * ngpus_per_node + gpu
    
    print("rank : %d" % rank)
    print("world_size : %d" % world_size)  #2
    
    distributed.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:7777',
                            world_size=world_size, 
                            rank=rank)
    
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][opts.backbone]
    dataset_cfg = cfg["dataset"][opts.dataset]
    if "mask_transformer" in opts.decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][opts.decoder]

    # model config
    if not opts.im_size:
        opts.im_size = dataset_cfg["im_size"]
    if not opts.crop_size:
        opts.crop_size = dataset_cfg.get("crop_size", opts.im_size)
    if not opts.window_size:
        opts.window_size = dataset_cfg.get("window_size", opts.im_size)
    if not opts.window_stride:
        opts.window_stride = dataset_cfg.get("window_stride", opts.im_size)

    model_cfg["image_size"] = (opts.crop_size, opts.crop_size)
    model_cfg["backbone"] = opts.backbone
    model_cfg["dropout"] = opts.dropout
    model_cfg["drop_path_rate"] = opts.drop_path
    decoder_cfg["name"] = opts.decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"] #8
    num_epochs = dataset_cfg["epochs"]
    lr = opts.lr
    
    if opts.eval_freq is None:
        opts.eval_freq = dataset_cfg.get("eval_freq", 1)

    if opts.normalization:
        model_cfg["normalization"] = opts.normalization

    # experiment config
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=opts.resume,
        dataset_kwargs=dict(
            dataset=opts.dataset,
            image_size=opts.im_size,
            crop_size=opts.crop_size,
            batch_size=world_batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=10,
        ),
        algorithm_kwargs=dict(
            batch_size=world_batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=opts.eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=opts.optimizer,
            lr=lr,
            weight_decay=opts.weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=opts.scheduler,
            epochs=num_epochs,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=opts.amp,
        # log_dir=opts.log_dir,
        inference_kwargs=dict(
            im_size=opts.im_size,
            window_size=opts.window_size,
            window_stride=opts.window_stride,
        ),
    )
  

    # dataset
    
    train_dst, val_dst, test_dst, n_classes, labels, labels_old, labels_cum = get_dataset(opts)
    
    train_loader = data.DataLoader(train_dst, batch_size=world_batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=world_batch_size if opts.crop_val else 1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    
    dataset_kwargs = variant["dataset_kwargs"]

    # model
    net_kwargs = variant["net_kwargs"]
    
    classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [16]..

    model = create_segmenter(net_kwargs, classes)
    
    logger.debug(model)
    
    if opts.step == 0:
        model_old = None
    else:
        old_classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step-1)
        model_old = create_segmenter(net_kwargs, old_classes)

        
    if model_old is not None:
        model = model.cuda(gpu)
        model_old = model_old.cuda(gpu)
        model_old = DDP(module=model_old, device_ids=[gpu], find_unused_parameters=True)

    else :
        model = model.cuda(gpu)
    
    model = DDP(module=model, device_ids=[gpu], find_unused_parameters=True)
        

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)
    # lr_scheduler = create_scheduler(opt_args, optimizer)
    lr_scheduler = PolyLR(optimizer, max_iters=num_epochs * len(train_loader), power=opts.lr_power)
    num_iterations = 0
    amp_autocast = suppress
    loss_scaler = None
    if opts.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        
    #load old model
    if opts.step > 0:
        path = f"{paths}/{set}/{task_name}_{opts.step - 1}.pth"
    
        if os.path.exists(path):
            step_checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model_old.load_state_dict(step_checkpoint['model_state'], strict=False)

            step_checkpoints = expand_class_embeddings(step_checkpoint, classes, gpu)
            
            model.load_state_dict(step_checkpoints['model_state'], strict=False)
            
            del step_checkpoint['model_state']
        else:
            raise FileNotFoundError(path)
            
        for par in model_old.parameters():
            par.requires_grad = False
        model_old.eval()

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module     



    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")
    
    trainer = Trainer(opts, model, model_old, gpu, opts.step, classes)
    metrics = StreamSegMetrics(n_classes)
    results = {}
    
    best_score = 0.0
    
    path2 = f"{paths}/{set}/{task_name}_{opts.step}.pth"
    if os.path.exists(path2):
        checkpoint = torch.load(path2, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        # optimizer.load_state_dict(checkpoint["optimizer_state"])
        # lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        best_score = checkpoint['best_score']
        
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    
    trainer.before(data_loader = train_loader)
    
    logger.add_table("Opts", vars(opts))

    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train_logger, tot_loss, loss, fod_loss, bm_loss = trainer.train_one_epoch(
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
        )
        print(train_logger)
        
        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("Total_Loss",tot_loss, epoch)
        logger.add_scalar("CE_Loss",loss, epoch)
        logger.add_scalar("FOD_Loss",fod_loss, epoch)
        logger.add_scalar("BM_Loss",bm_loss, epoch)

        if (epoch+1) % eval_freq == 0:
            val_score = trainer.evaluate(
                val_loader,
                metrics,
                opts.window_size,
                opts.window_stride,
                amp_autocast,
            )
            metrics.synch(gpu)


            # log stats
            if rank == 0:
                score = val_score['Mean IoU']
                if score > best_score:
                    best_score = score
                    save_ckpt(f"{paths}/{set}/{task_name}_{opts.step}.pth",
                            model, model_without_ddp, optimizer, lr_scheduler, epoch, score)

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], epoch)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], epoch)
            
            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

    # if rank == 0 :
    #     save_ckpt(f"/home/nayoung/nayoung/implement/Incrementer/checkpoints/disjoint/{task_name}_{opts.step}.pth",
    #                 model, model_without_ddp, optimizer, lr_scheduler, epoch, score)

    distributed.barrier()
    
    #test
    test_loader = data.DataLoader(test_dst, batch_size=world_batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)


    print(f"Test dataset length: {len(test_loader.dataset)}")
    classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
    model = create_segmenter(net_kwargs, classes)
    model = model.cuda(gpu)
    model = DDP(module=model, device_ids=[gpu], find_unused_parameters=True)
    ckpt = f"{paths}/{set}/{task_name}_{opts.step}.pth"
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    
    trainer = Trainer(opts, model, None, gpu, opts.step, classes=None)
    
    model.eval()
    
    test_score = trainer.evaluate(
                test_loader,
                metrics,
                opts.window_size,
                opts.window_stride,
                amp_autocast,
            )
    
    logger.add_table("Test_Class_IoU", test_score['Class IoU'])
    logger.add_table("Test_Class_Acc", test_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", test_score['Confusion Matrix'])
    results["T-IoU"] = test_score['Class IoU']
    results["T-Acc"] = test_score['Class Acc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", test_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", test_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", test_score['Mean Acc'], opts.step)

    logger.close()
    
    metrics.synch(gpu)
    if torch.distributed.get_rank() == 0:
        print('----test----')
        print(metrics.to_str(test_score))
    

if __name__ == "__main__":
    parser = argparsers.get_argparser()

    opts = parser.parse_args()
    opts = argparsers.modify_command_options(opts)
    
    world_size = 4
   
    mp.spawn(main, nprocs=4, args=(opts, 4, world_size))
