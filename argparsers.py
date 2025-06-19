import argparse
import tasks

def modify_command_options(opts):
    if opts.dataset == 'voc':
        opts.num_classes = 21
    if opts.dataset == 'ade':
        opts.num_classes = 150
    
    opts.no_overlap = not opts.overlap
    opts.no_cross_val = not opts.cross_val

    return opts

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='/scratch/kona419/Incrementer/VOCdevkit/',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'ade'], help='Name of dataset')
    parser.add_argument("--im_size", type=int, default=None, help="dataset resize size")
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_384")
    parser.add_argument("--decoder", type=str, default="mask_transformer")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--scheduler", type=str, default="polynomial")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--normalization", type=str, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    parser.add_argument("--task", type=str, default="19-1", choices=tasks.get_task_list(),
                        help="Task to be executed (default: 19-1)")
    parser.add_argument("--step", type=int, default=0,
                        help="The incremental step in execution (default: 0)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    
    parser.add_argument("--val_on_trainset", action='store_true', default=False,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--cross_val", action='store_true', default=False,
                        help="If validate on training or on validation (default: Train)")
    parser.add_argument("--crop_val", action='store_false', default=True,
                        help='do crop for validation (default: True)')
    
    parser.add_argument("--num_workers", type=int, default=1,
                        help='number of workers (default: 1)')
    parser.add_argument("--overlap", action='store_true', default=False,
                        help="Use this to not use the new classes in the old training set")
    parser.add_argument("--no_mask", action='store_true', default=False,
                        help="Use this to not mask the old classes in new training set")
    
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    
    parser.add_argument("--pseudo", type=str, default='entropy')
    parser.add_argument("--threshold", type=float, default=0.001)

    # Logging Options
    parser.add_argument("--logdir", type=str, default='logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=2,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument("--debug",  action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--visualize",  action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="epoch interval for saving model (default: 1)")

    return parser
