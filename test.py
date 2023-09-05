import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as albu
import numpy as np


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, trainsize, phase='train'):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)

        self.phase = phase

        self.transform = albu.Compose(
            [
                # TODO: リサイズ以外の処理を入れたいときはAlbumentationがおすすめ
            ]
        )

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.phase == 'train':
            image = np.array(image)
            gt = np.array(gt)

            # augmented = self.transform(image=image, mask=gt)
            # image, gt = augmented['image'], augmented['mask']

            image = Image.fromarray(image)
            gt = Image.fromarray(gt)
            image = image.convert('RGB')
            gt = gt.convert('L')

        image = self.img_transform(image)

        gt = self.gt_transform(gt)
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return self.size
