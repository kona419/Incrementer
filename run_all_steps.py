import os

command_template0 = "python /scratch/kona419/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.001 --overlap"
command_template1 = "python /scratch/kona419/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.0005 --overlap"
command_template23 = "python /scratch/kona419/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.0001 --overlap"
command_template45 = "python /scratch/kona419/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.00005 --overlap"

for step in range(0, 1):
    os.system(command_template0.format(step))

for step in range(1, 2):
    os.system(command_template1.format(step))

for step in range(2, 4):
    os.system(command_template23.format(step))

for step in range(4, 6):
    os.system(command_template45.format(step))
