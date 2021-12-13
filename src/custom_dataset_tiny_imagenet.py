from subset import DatasetFromSubset
from subset import class_names
from subset import download_image
from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile
import requests
from io import StringIO,BytesIO


class TinyImageNet(Dataset):
    def __init__(self,classes,url):
        self.data = []
        self.target = []
        self.classes = classes
        self.url = url
        
        wnids = open(f"{url}/wnids.txt", "r")
        
        for wclass in notebook.tqdm(wnids,desc='Loading Train Folder', total = 200):
          wclass = wclass.strip()
          for i in os.listdir(url+'/train/'+wclass+'/images/'):
            img = Image.open(url+"/train/"+wclass+"/images/"+i)
            npimg = np.asarray(img)
            
            if(len(npimg.shape) ==2):
             
               npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
            self.data.append(npimg)  
            self.target.append(self.classes.index(wclass))

        val_file = open(f"{url}/val/val_annotations.txt", "r")
        for i in notebook.tqdm(val_file,desc='Loading Test Folder',total =10000 ):
          split_img, split_class = i.strip().split("\t")[:2]
          img = Image.open(f"{url}/val/images/{split_img}")
          npimg = np.asarray(img)
          if(len(npimg.shape) ==2):
                    
                npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
          self.data.append(npimg)  
          self.target.append(self.classes.index(split_class))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        img = data     
        return data,target
      
  def TinyImageNetDataSet(train_split = 70,test_transforms = None,train_transforms = None):
    down_url  = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    download_images(down_url)
    classes = class_names(url = "tiny-imagenet-200/wnids.txt")
    dataset = TinyImageNet(classes,url="tiny-imagenet-200")
    train_len = len(dataset)*train_split//100
    test_len = len(dataset) - train_len 
    train_set, val_set = random_split(dataset, [train_len, test_len])
    train_dataset = DatasetFromSubset(train_set, transform=train_transforms)
    test_dataset = DatasetFromSubset(val_set, transform=test_transforms)
    return train_dataset, test_dataset,classes
