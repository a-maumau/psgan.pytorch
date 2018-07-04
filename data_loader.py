import os
import random

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from PIL import Image

from tqdm import tqdm

def print_error(e):
    import traceback
    traceback.print_exc()
    print(e)

class DTDDataLoader(data.Dataset):
    def __init__(self, data_root, img_list_path="train.txt", transform=None):
        """
            args:
                data_root: str
                    root directory of the dataset.
                    if we are using DTD, it should be the "path_to_DTDdataset_file/dtd/images"

                img_list_path: str
                    list for which texture name to use, it should be per line like
                        dotted
                        stripe
                        blotchy
                        .
                        .
                        .

                transform: torchvision.transforms
                    image transfoms.
                    if it is None, it will perform, ToTensor->Normalize(std, mean at .5)
        """
        self.transform = transform
        self.images = []

        # read texture names
        with open(os.path.join(img_list_path), "r") as file:
            self.texture_names = file.readlines()
        self.texture_names = [name.rstrip("\n") for name in self.texture_names]

        tqdm.write("loading images...")

        # read the 
        for tex_name in tqdm(self.texture_names, desc="textures", ncols=80):
            texture_file_path = os.path.join(data_root, tex_name)
            try:
                images = os.listdir(texture_file_path)
            except Exception as e:
                #print_error(e)
                tqdm.write("pass {}".format(tex_name))

            for img_name in images:
                try:
                    self.images.append(Image.open(os.path.join(texture_file_path, img_name)).convert('RGB'))
                except Exception as e:
                    #print_error(e)
                    tqdm.write("pass {}".format(img_name))
                #break


        self.data_num = len(self.images)

    def __getitem__(self, index):
        """
            the single image is pick from the index.
        """

        if self.transform is not None:
            img = self.transform(self.images[index])
        else:
            img = transforms.ToTensor()(self.images[index])
            img = transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))(img)
                
        return img

    def __len__(self):
        return self.data_num

# for your own data, which read all file in data_dir
class ImageDataLoader(data.Dataset):
    def __init__(self, data_dir, transform=None):
        """
            args:
                data_root: str
                    root directory of the dataset.

                transform: torchvision.transforms
                    image transfoms.
                    if it is None, it will perform, ToTensor->Normalize(std, mean at .5)
        """
        self.transform = transform
        self.images = []

        tqdm.write("loading images...")

        images = os.listdir(data_dir)

        # read the 
        for img_name in tqdm(images, desc="image", ncols=80):
            img_file_path = os.path.join(data_dir, img_name)
            try:
                self.images.append(Image.open(img_file_path).convert('RGB'))
            except Exception as e:
                #print_error(e)
                tqdm.write("pass {}".format(img_name))

        self.data_num = len(self.images)

    def __getitem__(self, index):
        """
            the single image is pick from the index.
        """

        if self.transform is not None:
            img = self.transform(self.images[index])
        else:
            img = transforms.ToTensor()(self.images[index])
            img = transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))(img)
                
        return img

    def __len__(self):
        return self.data_num

# data loader for dataset
def get_loader(data_set, batch_size, shuffle, num_workers):
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return data_loader
