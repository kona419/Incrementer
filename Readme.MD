1. download the dataset with download_voc.sh
    It seems the dataset download link doesn't exits.
    Use the train_aug.txt file that I attached.
2. check the dataset directory and change the command template of argparsers.py
    parser.add_argument("--data_root", type=str, default='/scratch/kona419/Incrementer/VOCdevkit/',  help="path to Dataset")
3. command template in run_all_steps.py
    "python /home/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.001 --overlap"
    --dataset dataset directory
4. check all directory in run.sh and change into your own directory.

5. run run.sh