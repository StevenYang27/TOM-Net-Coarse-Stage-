import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from preprocess import TOMTransform, TRAIN_SIMPLE_2k
from utils import readFloFile


class TOMDataset(Dataset):
    def __init__(self, transforms, img_dir, img_list_file, inference=False):
        """
        Dataset class for TOM-NET Dataset
        :param transforms: torchvision.transforms; data augmentation
        :param img_dir: path_like string; directory for all images
        :param img_list_file: path_like string; txt file for all image file names
        :param inference: boolean; whether to transform input images
        """
        self.transforms = transforms
        '''
        The file tree of TOM-Net Dataset looks like this:
            Dataset
            |--Image Directory(contains all images)
            |--Sample_names.txt(contains sample image file names in Image Directory)
        '''
        self.img_dir = os.path.join('data', img_dir)  # image subdirectory
        self.img_list_file = os.path.join('data', img_list_file)  # txt file of image file names
        self.inference = inference
        with open(self.img_list_file) as infile:
            self.data_list = infile.read().splitlines()
        if self.data_list[-1] == '':
            self.data_list.pop()

    def __getitem__(self, idx):
        data_name = self.data_list[idx]  # 'xxx.jpg'
        file_name, _ = os.path.splitext(data_name)  # 'xxx'

        data_path = os.path.join(self.img_dir, data_name)  # 'usr/.../Dataset/Images Directory/xxx.jpg'
        img = Image.open(data_path)

        mask_name = file_name + '_mask.png'  # 'xxx_mask.png'
        mask_path = os.path.join(self.img_dir, mask_name)
        mask = Image.open(mask_path)

        rho_name = file_name + '_rho.png'
        rho_path = os.path.join(self.img_dir, rho_name)
        rho = Image.open(rho_path)

        flow_name = file_name + '_flow.flo'  # 'xxx_flo.flo'
        flow_path = os.path.join(self.img_dir, flow_name)  # 'usr/.../Dataset/Images Directory/xxx_flo.flo'
        flow = readFloFile(flow_path)

        ref_name = file_name + '_ref.jpg'
        ref_path = os.path.join(self.img_dir, ref_name)
        ref = Image.open(ref_path)

        labels = [mask, rho, flow, ref]

        if self.transforms is not None:
            rand = torch.rand(1)
            img = self.transforms(img, rand, is_data=True)
            labels = list(map(self.transforms, labels, [rand] * 4))
            labels.insert(-1, img)
        else:
            transform = T.ToTensor()
            img = transform(img)
            labels = tuple(map(transform, labels))

        if not self.inference:
            return img, labels
        else:
            return img, labels, data_name

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    EPOCHS = 2
    BATCH_SIZE = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = TOMTransform(train=True, dataset=TRAIN_SIMPLE_2k)
    TestDataset = TOMDataset(transforms, 'Images', 'img_list.txt')
    test_iter = DataLoader(TestDataset, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        for itr, (data, label) in enumerate(test_iter):
            print(f'Epoch: {epoch + 1} | Iteration: {itr + 1} | Image Size: {data.shape} | Mask Size: {label[0].shape} '
                  f'| Attenuation Size: {label[1].shape} | Flow Size: {label[2].shape} | Reference Size: {label[-1].shape}')
