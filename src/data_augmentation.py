from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="../data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

channel_means = (0.4919, 0.4827, 0.4472)
channel_stdevs = (0.2470, 0.2434, 0.2616)
def getTrainTransforms():
    train_transforms = A.Compose(
    [ #32(img size)+ 4(padding) along x and 4(padding) along y =40
      A.PadIfNeeded(min_height=40, min_width=40, border_mode = cv2.BORDER_REFLECT, always_apply=True),
      A.RandomCrop(height=32, width=32, always_apply=True),
      A.HorizontalFlip(p = 0.7),
      A.Normalize(mean=channel_means, std=channel_stdevs),
      A.Cutout(num_holes=1, max_h_size=16,max_w_size = 16,p=0.7)  ,                                                                            
      ToTensorV2()     
    ])
    return train_transforms

def getTestTransforms():
    test_transforms = A.Compose(
    [
     A.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233,0.24348505,0.26158768)),
     ToTensorV2(),
    ])

    return test_transforms