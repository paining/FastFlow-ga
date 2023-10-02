import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from image_crop import image_crop

import logging
logger = logging.getLogger(__name__)


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace('\\', '/').replace("/test/", "/ground_truth/").replace(".png", "_mask.png")
                )
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)

class GlotecDataset(torch.utils.data.Dataset):
    def __init__(self, root, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            file_list = []
            for dirpath, dirs, files in os.walk(root):
                dirs.sort()
                files.sort()
                file_list.extend([
                    os.path.join(dirpath, file) for file in files 
                    if file.endswith(".jpg") or file.endswith(".png")
                ])
            self.image_files = file_list
        else:
            file_list = []
            for dirpath, dirs, files in os.walk(root):
                dirs.sort()
                files.sort()
                file_list.extend([
                    os.path.join(dirpath, file) for file in files 
                    if file.endswith(".jpg") or file.endswith(".png")
                ])
            self.image_files = file_list
            #self.image_files = glob(os.path.join(root, category, "*.jpg"))
            # self.target_transform = transforms.Compose(
            #     [
            #         transforms.Resize(input_size),
            #         transforms.ToTensor(),
            #     ]
            # )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if self.is_train:
            return image
        # else:
        #     if os.path.dirname(image_file).endswith("good"):
        #         target = torch.zeros([1, image.shape[-2], image.shape[-1]])
        #     else:
        #         target = Image.open(
        #             image_file.replace('\\', '/').replace("/test/", "/ground_truth/").replace(".jpg", "_mask.png")
        #         )
        #         target = self.target_transform(target)
        #     return image, target
        return image, image_file

    def __len__(self):
        return len(self.image_files)

        
class DACDataset(torch.utils.data.Dataset):
    def __init__(self, root, input_size, is_train=True, crop=False):
        self.input_size = input_size
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            # self.image_files = []
            # for root, dirs, files in os.walk(root):
            #     dirs.sort()
            #     files.sort()
            #     self.image_files.extend([
            #         os.path.join(root, file)
            #         for file in files
            #         if file.endswith(".bmp")
            #     ])
            self.image_files = glob(
                os.path.join(root, "train", "good", "*.bmp")
            )
        else:
            self.image_files = glob(os.path.join(root, "test", "*", "*.bmp"))
            self.target_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train
        self.crop = crop

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert('RGB')
        image = image.crop((0, 0, self.input_size[0], self.input_size[1]))

        if self.crop:
            l, r, w = image_crop(image, 50)
            cropsize = 32 * (w // 32)
            offset = (w % 32) // 2
            l = l + offset
            image = image.crop((l, 0, l+cropsize, 416))

        # l, r, w = image_crop(np.asarray(image), 30)
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                try:
                    target = Image.open(
                        image_file.replace('\\', '/').replace("/test/", "/ground_truth/").replace(".bmp", ".png")
                    ).convert('1')
                    target = target.crop((0, 0, self.input_size[0], self.input_size[1]))
                    if self.crop:
                        target = target.crop((l, 0, l+cropsize, 416))
                    target = self.target_transform(target)
                except FileNotFoundError as e:
                    logger.exception(e)
                    target = torch.zeros_like(image, dtype=torch.int32)
            return image, target, image_file
            # return image, image_file

    def __len__(self):
        return len(self.image_files)



class SKONDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size[::-1]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.jpg")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.jpg"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = 0
            else:
                target = 1
            return image, target, image_file

    def __len__(self):
        return len(self.image_files)