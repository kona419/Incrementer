import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import MetricLogger
from metrics import gather_data, compute_metrics
from model import utils

from utils.loss import visFODistill, clsFODistill, BMloss
# import segm.utils.torch as ptu
from functools import reduce

def resize_targets_for_patches(targets, patch_size, output_size):
    """
    targets: [batch_size, height, width]
    patch_size: 패치의 크기 (예: 16)
    output_size: 출력 텐서의 공간적 차원 (예: [32, 32])
    """
    targets = targets.unsqueeze(1)

    targets_resized = F.interpolate(targets.float(), size=output_size, mode='nearest')
    
    return targets_resized.squeeze(1).long()

class Trainer:
    def __init__(self, opts, model, model_old, gpu, step, classes=None):
        self.model_old = model_old
        self.model = model
        self.gpu = gpu
        self.step = step
        
        self.pseudo_labeling = opts.pseudo
        self.threshold = opts.threshold
        
        # print(classes)
        
        if classes is not None:
            self.tot_classes = reduce(lambda a, b: a + b, classes)
            self.new_classes = classes[-1]
            self.old_classes = self.tot_classes - self.new_classes
        else:
            self.old_classes = 0

        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.vis_fod = visFODistill()
        self.cls_fod = clsFODistill()
        self.bm_loss = BMloss()
        
    def before(self, data_loader):
        if self.pseudo_labeling is None:
            return
        if self.pseudo_labeling.split("_")[0] == "median" and self.step > 0:
            self.thresholds, _ = self.find_median(data_loader)
        elif self.pseudo_labeling.split("_")[0] == "entropy" and self.step > 0:
            self.thresholds, self.max_entropy = self.find_median(
                data_loader, mode="entropy"
            )
        
    def train_one_epoch(self,
        data_loader,
        optimizer,
        lr_scheduler,
        epoch,
        amp_autocast,
    ): 
        
        logger = MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}]"
        print_freq = 100
        lambda_val = 1.0

        self.model.train()
        data_loader.sampler.set_epoch(epoch)
        num_updates = epoch * len(data_loader)
        for (samples, targets) in logger.log_every(data_loader, print_freq, header):

            samples = samples.to(self.gpu, dtype=torch.float32, non_blocking=True)
            targets = targets.to(self.gpu, dtype=torch.long, non_blocking=True)
            
            if (self.pseudo_labeling is not None) and self.model_old is not None:
                with torch.no_grad():
                    output_old = self.model_old.forward(samples)
                    vis_emb_old = self.model_old.module.decoder.patches
                    cls_emb_old = self.model_old.module.decoder.cls_seg_feat


                    
            with amp_autocast():
                output = self.model.forward(samples)
                vis_emb = self.model.module.decoder.patches
                cls_emb = self.model.module.decoder.cls_seg_feat
                
            classif_adaptive_factor = 1.0
            if self.step > 0:
                mask_background = targets < self.old_classes
                
                targets[mask_background] = output_old.argmax(dim=1)[mask_background]
               
                #FOD_vis                
                alpha = torch.zeros_like(targets, dtype=torch.float)
                alpha[~mask_background] = 0
                alpha[mask_background] = 1
                alpha[targets == 0] = self.old_classes / self.tot_classes
                
                _, _, H_p, W_p = vis_emb.size()
                
                alpha_resized = F.interpolate(alpha.unsqueeze(1).float(), size=(H_p, W_p), mode='bilinear', align_corners=False).squeeze(1)
                
                fod_vis_loss = self.vis_fod(vis_emb, vis_emb_old, alpha_resized)
                
                beta = torch.ones(self.old_classes)
                beta[0] = self.old_classes / self.tot_classes
                beta = beta.to(self.gpu)
                
                fod_cls_loss = self.cls_fod(cls_emb, cls_emb_old, beta, self.old_classes)

                #L_BM 
                new_classes_mask = (targets.unsqueeze(1) == self.new_classes).any(dim=1)
                
                B_i = torch.zeros_like(targets, dtype=torch.float)
                B_i[new_classes_mask] = 1.0
                
                bm_loss = self.bm_loss(output, output_old, B_i, self.old_classes)
                
                weights = torch.ones_like(targets, dtype=torch.float)
                weights[new_classes_mask] = lambda_val * math.sqrt(self.new_classes / self.tot_classes)
                
            # print(self.new_classes)
            # print(self.tot_classes)
            # print(targets.size())
                
            
            with amp_autocast():
                loss = self.criterion(output, targets)
   
            if self.step > 0:
                with amp_autocast():
                    # weights = torch.ones_like(targets, dtype=torch.float)
                    # for new_class in self.new_classes:
                    #     weights[targets == new_class] = self.lambda_val * math.sqrt(self.new_classes / self.tot_classes)
                    loss = (loss * weights).mean()
                loss_tot = loss + fod_vis_loss + fod_cls_loss + bm_loss
            else:
                loss = loss.mean()
                loss_tot = loss
                fod_vis_loss = 0
                fod_cls_loss = 0
                bm_loss = 0

            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()

            num_updates += 1
            
            lr_scheduler.step()

            torch.cuda.synchronize()

            logger.update(
                loss_tot=loss_tot.item(),
                learning_rate=optimizer.param_groups[0]["lr"],
            )

        return logger, loss_tot.item(), loss, fod_vis_loss+fod_cls_loss, bm_loss
    
    def find_median(self, train_loader, mode="probability"):
        """Find the median prediction score per class with the old model.

        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.

        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        if mode == "entropy":
            max_value = torch.log(torch.tensor(self.tot_classes).float().to(self.gpu))
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]

        histograms = torch.zeros(self.tot_classes, nb_bins).long().to(self.gpu)
        
        logger = MetricLogger(delimiter="  ")
        header = "pseudo_label:"
        print_freq = 100

        for (samples, targets) in logger.log_every(train_loader, 100, header):
            samples = samples.to(self.gpu, dtype=torch.float32, non_blocking=True)
            targets = targets.to(self.gpu, dtype=torch.long, non_blocking=True)

            outputs_old = self.model_old.forward(samples)

            mask_bg = targets == 0
            probas = torch.softmax(outputs_old, dim=1)
            max_probas, pseudo_labels = probas.max(dim=1)

            if mode == "entropy":
                values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
                accumulate=True
            )

        thresholds = torch.zeros(self.tot_classes, dtype=torch.float32).to(
            self.gpu
        )  # zeros or ones? If old_model never predict a class it may be important

        for c in range(self.tot_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue

            half = total / 2
            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = self.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)


        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)

        return thresholds.to(self.gpu), max_value


    @torch.no_grad()
    def evaluate(self,
        data_loader,
        metrics,
        window_size,
        window_stride,
        amp_autocast,
    ):
        metrics.reset()
        
        model_without_ddp = self.model
        if hasattr(self.model, "module"):
            model_without_ddp = self.model.module
        logger = MetricLogger(delimiter="  ")
        header = "Eval:"
        print_freq = 50

        val_seg_pred = {}
        self.model.eval()
        class_count = 0
        for images, target in logger.log_every(data_loader, print_freq, header):
            images = images.to(self.gpu, dtype=torch.float32, non_blocking=True)
            target = target.to(self.gpu, dtype=torch.long, non_blocking=True)
        
                    
            with amp_autocast():
                output = self.model.forward(images)
            _, output = output.max(dim=1)
                
            target = target.cpu().numpy()
            output = output.cpu().numpy()
            metrics.update(target,output)
            
        # print(class_count)
        
        metrics.synch(self.gpu)
        score = metrics.get_results()

        return score
    
def entropy(probabilities):
    """Computes the entropy per pixel.

    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
        Saporta et al.
        CVPR Workshop 2020

    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)
