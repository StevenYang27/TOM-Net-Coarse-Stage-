import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCELoss, MSELoss

from dataloader import DataLoader, TOMDataset
from preprocess import TOMTransform, DATASET_STAT
from utils import render


class MaskLoss(object):
    def __init__(self, alpha=0.1):
        """
        object mask segmentation loss(BCE)
        :param alpha: float; coefficient
        """
        self.alpha = alpha

    def __call__(self, pred, label):
        """
        object mask segmentation loss(BCE)
        :param pred: shape: [B, 2, H, W]; without softmax
        :param label: shape: [B, 1, H, W]; after normalization
        :return: loss
        """
        n, c, h, w = pred.size()
        pred = torch.softmax(pred, dim=1)
        label = F.one_hot(label.long(), num_classes=c).float()  # one-hot encoding -> [B, 1, H, W, 2]
        label = torch.squeeze(label, dim=1)  # [B, 1, H, W, 2] -> [B, H, W, 2]
        label = label.permute(0, 3, 1, 2)  # [B, H, W, 2] -> [B, 2, H, W]
        criterion = BCELoss()
        loss = criterion(pred, label) * self.alpha
        return loss


class AttenuationLoss(object):
    def __init__(self, alpha=1):
        """
        attenuation regression loss(MSE)
        :param alpha: float; coefficient
        """
        self.alpha = alpha

    def __call__(self, pred, label):
        """
        attenuation regression loss(MSE)
        :param pred:
        :param label:
        :return:
        """
        n, c, h, w = pred.size()
        pred = torch.sigmoid(pred)
        criterion = MSELoss()
        loss = criterion(pred, label) * self.alpha
        return loss


class FlowLoss(object):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, pred, label):
        n, c, h, w = pred.size()
        res = torch.tanh(pred)
        pred *= w
        label *= w
        loss = torch.norm(label - res, dim=1, p=2).mean() * self.alpha
        return loss


class ReconstructionLoss(object):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def __call__(self, pred, label):
        n, c, h, w = pred.size()
        criterion = MSELoss()
        loss = criterion(pred, label) * self.alpha
        return loss


class TOMLoss(object):
    """
    Total loss for coarse stage
    """
    def __init__(self, ms=0.1, ar=1, fr=0.01, ir=1):
        """
        total loss for coarse stage
        :param ms: float; coefficient for mask segmentation loss
        :param ar: float; coefficient for attenuation regression loss
        :param fr: float; coefficient for flow regression loss
        :param ir: float; coefficient for image reconstruction loss
        """
        self.mask_alpha = ms
        self.attenuation_alpha = ar
        self.flow_alpha = fr
        self.reconstruction_alpha = ir

        self.mask_criterion = MaskLoss(alpha=self.mask_alpha)
        self.attenuation_criterion = AttenuationLoss(alpha=self.attenuation_alpha)
        self.flow_criterion = FlowLoss(alpha=self.flow_alpha)
        self.reconstruction_criterion = ReconstructionLoss(alpha=self.reconstruction_alpha)

    def __call__(self, pred, label):
        mask_loss = 0
        attenuation_loss = 0
        flow_loss = 0
        reconstruction_loss = 0

        scales = list(pred.keys())
        for scale in scales:
            ratio = 1 / np.power(2, 4 - scale)  # multi-scale loss
            for i in range(len(label)):
                label[i] = F.interpolate(label[i], scale_factor=0.5 if scale != 1 else 1)
            reconstruction_img = render(pred[scale][0], pred[scale][1], pred[scale][2], label[4])

            mask_loss += self.mask_criterion(pred[scale][0], label[0]) * ratio
            attenuation_loss += self.attenuation_criterion(pred[scale][1], label[1]) * ratio
            flow_loss += self.flow_criterion(pred[scale][2], label[2]) * ratio
            reconstruction_loss += self.reconstruction_criterion(reconstruction_img, label[3]) * ratio

        total_loss = mask_loss + attenuation_loss + flow_loss + reconstruction_loss

        return total_loss


def main():
    scales = [1, 2, 3, 4]
    ratio = [1 / np.power(2, scale - 1) for scale in scales]
    masks = [torch.from_numpy(np.random.rand(2, 2, int(512 * ratio[i]), int(512 * ratio[i]))).float() for i in range(4)]
    attenuations = [torch.from_numpy(np.random.rand(2, 1, int(512 * ratio[i]), int(512 * ratio[i]))).float() for i in
                    range(4)]
    flows = masks
    preds = {scales[0]: [masks[0], attenuations[0], flows[0]],
             scales[1]: [masks[1], attenuations[1], flows[1]],
             scales[2]: [masks[2], attenuations[2], flows[2]],
             scales[3]: [masks[3], attenuations[3], flows[3]]}

    BATCH_SIZE = 2
    transforms = TOMTransform(train=True, dataset=DATASET_STAT['simple'])
    TestDataset = TOMDataset(transforms, 'Images', 'img_list.txt')
    test_iter = DataLoader(TestDataset, batch_size=BATCH_SIZE)
    for itr, (data, label) in enumerate(test_iter):
        labels = label
        break

    criterion = TOMLoss()
    loss = criterion(preds, labels)
    print(loss)


if __name__ == '__main__':
    main()
