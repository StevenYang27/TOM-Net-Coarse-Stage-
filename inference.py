import argparse
import os

import torch
from torchvision import transforms as T

from dataloader import DataLoader, TOMDataset, TOMTransform
from preprocess import DATASET_STAT
from utils import render

parser = argparse.ArgumentParser()
parser.add_argument('--root_directory', type=str, default='data')
parser.add_argument('--image_folder', type=str, default='Images')
parser.add_argument('--image_list_file', type=str, default='img_list.txt')
parser.add_argument('--dataset', type=str, default='simple', choices=['simple', 'complete'])
parser.add_argument('--model_path', type=str, default='./output/Epoch-1.pt')
parser.add_argument('--result_folder', type=str, default='./result')


args = parser.parse_args()


def inference(dataloader, model, device):
    model.to(device)
    model.eval()
    res = {}
    for _, (data, label, name) in enumerate(dataloader):
        data.to(device)
        label = [l.to(device) for l in label]

        n = data.size()[0]  # batch size

        pred = model(data)[1]  # original size
        final_image = render(mask=pred[0],
                             rho=pred[1],
                             flow=pred[2],
                             ref=label[-1])
        final_image = torch.squeeze(final_image, dim=0)
        pred[0] = torch.argmax(pred[0], dim=1, keepdim=True).to(torch.float32)
        outputs = pred[:2]
        transform = T.ToPILImage()
        for i in range(n):
            img = transform(final_image[i])
            out = list(map(transform, pred))
            res.update({name[i]: img})
    return res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = TOMTransform(train=False, dataset=DATASET_STAT[args.dataset])
    dataset = TOMDataset(transforms=transform, img_dir=args.image_folder, img_list_file=args.image_list_file,
                         inference=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2)
    model = torch.load(args.model_path)
    pred = inference(dataloader, model, device)
    for (img_name, img) in pred.items():
        img_path = os.path.join(args.result_folder, img_name)
        img.save(img_path)


if __name__ == '__main__':
    main()
