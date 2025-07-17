## This is unofficial code of "Incrementer: Transformer for Class-Incremental Semantic Segmentation with Knowledge Distillation Focusing on Old Class"

## How to run

1. download the dataset with download_voc.sh

    It seems the dataset download link doesn't exits.

    Use the train_aug.txt file that I attached.

2. check the dataset directory and change the command template of argparsers.py
```
    parser.add_argument("--data_root", type=str, default='/scratch/kona419/Incrementer/VOCdevkit/',  help="path to Dataset")
```
3. command template in run_all_steps.py
```
    "python /home/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.001 --overlap"
    --dataset dataset directory
```
4. check all directory in run.sh and change into your own directory.

5. run 
```run.sh```

## Result
<img width="1608" height="187" alt="Image" src="https://github.com/user-attachments/assets/e0b24e89-6344-4191-9de7-35f93f9656a9" />
<img width="900" height="100" alt="Image" src="https://github.com/user-attachments/assets/f829caf6-c132-410d-afbe-ad66543c14dc" />


## BibTex
```
@inproceedings{shang2023incrementer,
  title={Incrementer: Transformer for class-incremental semantic segmentation with knowledge distillation focusing on old class},
  author={Shang, Chao and Li, Hongliang and Meng, Fanman and Wu, Qingbo and Qiu, Heqian and Wang, Lanxiao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7214--7224},
  year={2023}
}
```

## Acknowlegements

This code is based on [Segmenter](https://github.com/rstrudel/segmenter) library and the semantic segmentation training and evaluation pipeline 
is using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
