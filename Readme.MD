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

| Method | Frame | \| 19-1 (2 steps) — Disjoint \| |     |     | \| 19-1 (2 steps) — Overlapped \| |     |     | \| 15-5 (2 steps) — Disjoint \| |     |     | \| 15-5 (2 steps) — Overlapped \| |     |     | \| 15-1 (6 steps) — Disjoint \| |     |     | \| 15-1 (6 steps) — Overlapped \| |     |     |
|--------|--------|--------|--------|--------|----------|--------|--------|--------|--------|--------|-----------|--------|--------|--------|--------|--------|-----------|--------|--------|
|        |        | 1-19   | 20     | all    | 1-19     | 20     | all    | 1-15   | 16-20  | all    | 1-15      | 16-20  | all    | 1-15   | 16-20  | all     | 1-15   | 16-20  | all    |
| EWC† [20] |        | 23.20  | 16.00  | 22.90  | 26.90    | 14.00  | 26.30  | 26.70  | 37.70  | 29.40  | 24.30     | 35.50  | 27.10  | 0.30   | 4.30   | 1.30    | 0.30   | 4.30   | 1.30   |


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
