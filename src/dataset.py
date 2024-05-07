import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class TargetedDataset(Dataset):

    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):

        img_list = os.listdir(self.img_dir)
        img_path = os.path.join(self.img_dir, img_list[idx])
        image = read_image(img_path).float() / 255
        label = int(img_list[idx].split("_")[0])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, img_list[idx]
    

class UniversalDataset(Dataset):

    def __init__(self, img_dir, eval=False, transform=None, target_transform=None):
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.eval = eval

    def __len__(self):
        return int(len(os.listdir(self.img_dir)) * (2 / 5))

    def __getitem__(self, idx):

        img_list = os.listdir(self.img_dir)
        new_img_list = []
        for img in img_list:
            if img.split("_")[1] == '0.png' or img.split("_")[1] == '1.png':
                new_img_list.append(img)
        img_list = new_img_list

        img_path = os.path.join(self.img_dir, img_list[idx])
        image = read_image(img_path).float() / 255
        if self.eval:
            image += (read_image("universal.png").float() - 12) / 255
        
        label = int(img_list[idx].split("_")[0])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label