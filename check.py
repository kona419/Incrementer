import torch

def load_epoch_from_checkpoint(path):
    """ load the saved model and return the epoch number
    """
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    return epoch

# 예를 들어, 모델이 'model.pth'에 저장되어 있다고 가정
path = 'model.pth'
loaded_epoch = load_epoch_from_checkpoint('/home/nayoung/nayoung/implement/Incrementer/checkpoints/disjoint/15-5-voc_0.pth')
print(f"Loaded epoch: {loaded_epoch}")
