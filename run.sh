#!/bin/sh
#SBATCH --job-name=miniLLM 
#SBATCH --gres=gpu:v100l:4 
#SBATCH --qos=normal 
#SBATCH --time=10:00:00 
#SBATCH -c 16 
#SBATCH --mem=16G 
#SBATCH --output=/scratch/kona419/Incrementer/result/15-1/15-1.out 
#SBATCH --error=/scratch/kona419/Incrementer/result/15-1/error/15-1.out 

export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/11.8.0 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/nix/store/z87lf4q1l809fpnmsj9850nb5qxvw2lv-glog-0.3.4/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/nix/store/xp5ybf0b5mwrc098padfh9av4az94g7q-libtiff-4.0.7/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/gcc9/opencv/4.8.0/lib/python3.10/site-packages:$PYTHONPATH

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 기본 600 → 1800으로 증가
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1


source /scratch/kona419/Incrementer/incrementer/bin/activate

python /scratch/kona419/Incrementer/run_all_steps.py
