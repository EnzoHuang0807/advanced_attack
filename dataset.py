import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):

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