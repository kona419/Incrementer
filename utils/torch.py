import os
import torch


"""
GPU wrappers
"""

use_gpu = False
gpu_id = 7
device = None

distributed = False
dist_rank = 0
world_size = 1




def set_gpu_mode(mode):
    # global use_gpu
    # global device
    # global gpu_id
    # global distributed
    # global dist_rank
    # global world_size
    # gpu_id = int(os.environ.get("SLURM_LOCALID", 0))
    # dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    # world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]= "7" 
    
    use_gpu = True
    gpu_id = 7
    device = None

    distributed = True
    dist_rank = 0
    world_size = 1

    distributed = world_size > 1
    use_gpu = mode
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
