import argparse
import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from CoarseNet import CoarseNet
from dataloader import DataLoader, TOMDataset
from loss import TOMLoss
from preprocess import TOMTransform, DATASET_STAT
from utils import AverageMeter, init_weights

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--root_directory', type=str, default='data')
parser.add_argument('--image_folder', type=str, default='Images')
parser.add_argument('--image_list_file', type=str, default='img_list.txt')
parser.add_argument('--validation_list_file', type=str, default='img_list.txt')
parser.add_argument('--dataset', type=str, default='simple', choices=['simple', 'complete'])
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=1)

args = parser.parse_args()


def train(dataloader, model, device, criterion, optimizer, epoch, total_batch):
    model.to(device)
    model.train()
    loss_meter = AverageMeter()
    with tqdm(dataloader, unit='batch') as tepoch:
        for batch_id, (data, label) in enumerate(tepoch):
            tepoch.set_description(f'Epoch {epoch}')
            data = data.to(device)
            label = [l.to(device) for l in label]

            optimizer.zero_grad()

            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            n = data.size()[0]
            loss_meter.update(loss.detach().numpy(), n)

            tepoch.set_postfix_str(f'Batch Loss: {loss.detach().numpy():.4f}')

            # print(f'Epoch[{epoch:03d}/{args.num_epochs:03d}], ' +
            #       f'Step[{batch_id:04d}/{total_batch:04d}], ' +
            #       f'Batch Loss: {loss.detach().numpy():.4f}')

    return loss_meter.avg


def validate(dataloader, model, device, criterion):
    model.to(device)
    model.eval()
    loss_meter = AverageMeter()
    for _, (data, label) in enumerate(dataloader):
        data = data.to(device)
        label = [l.to(device) for l in label]

        pred = model(data)
        loss = criterion(pred, label)

        n = data.size()[0]
        loss_meter.update(loss.detach().numpy(), n)

    return loss_meter.avg


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = TOMDataset(transforms=TOMTransform(train=True, dataset=DATASET_STAT[args.dataset]),
                               img_dir=args.image_folder,
                               img_list_file=args.image_list_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
    val_dataset = TOMDataset(transforms=TOMTransform(train=False, dataset=DATASET_STAT[args.dataset]),
                             img_dir=args.image_folder,
                             img_list_file=args.image_list_file)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size)
    total_batch = int(len(train_dataset) / args.batch_size)

    model = CoarseNet()
    model.apply(init_weights)
    criterion = TOMLoss()
    optimizer = Adam(params=model.parameters(),
                     lr=args.lr)
    scheduler = StepLR(optimizer=optimizer,
                       step_size=3)

    with torch.autograd.set_detect_anomaly(True):

        for epoch in range(1, args.num_epochs + 1):
            train_loss = train(dataloader=train_dataloader,
                               model=model,
                               device=device,
                               criterion=criterion,
                               optimizer=optimizer,
                               epoch=epoch,
                               total_batch=total_batch)
            val_loss = validate(dataloader=val_dataloader,
                                model=model,
                                device=device,
                                criterion=criterion)
            scheduler.step()

            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f} Validate Loss: {val_loss:.4f}")

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f'Epoch-{epoch}.pt')
                torch.save(model, model_path)
                print(f'----- Save model: {model_path}')


if __name__ == '__main__':
    main()
