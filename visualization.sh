import os

# Command template
command_template0 = "python /home/nayoung/nayoung/Incrementer3/segm/visual.py --task 15-5s --step {} --overlap"
# command_template1 = "python /home/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.0005 --overlap"
# command_template23 = "python /home/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.0001 --overlap"
# command_template45 = "python /home/Incrementer/segm/train.py --dataset voc --task 15-5s --step {} --lr 0.00005 --overlap"


# Loop through steps 1 to 5
for step in range(0, 6):
    # Construct the command with the current step
    command = command_template0.format(step)
    # Execute the command
    os.system(command)

# for step in range(1, 2):
#     # Construct the command with the current step
#     command = command_template1.format(step)
#     # Execute the command
#     os.system(command)

# for step in range(2, 4):
#     # Construct the command with the current step
#     command = command_template23.format(step)
#     # Execute the command
#     os.system(command)

# for step in range(4, 6):
#     # Construct the command with the current step
#     command = command_template45.format(step)
#     # Execute the command
#     os.system(command)
