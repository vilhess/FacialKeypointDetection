import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import ToTensor


class FacialKeyPoint(Dataset):
    def __init__(self, transform = ToTensor()):
        self.df = pd.read_json("data/all_data.json")
        self.transform = transform

    def __len__(self):
        return len(self.df.columns)
    
    def __getitem__(self, index):
        img_path = os.path.join('data/images/', self.df[index]['file_name'])
        nose_pos = self.df[index]['face_landmarks'][30]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, torch.Tensor(nose_pos)