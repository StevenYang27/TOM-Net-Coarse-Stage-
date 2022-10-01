import cv2
import matplotlib
import numpy as np
import torch
import torchvision.transforms as T
from torch.nn import Conv2d


def readFloFile(filename, short=True):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise Exception('Magic number incorrect: %s' % filename)
        w = int(np.fromfile(f, np.int32, count=1))
        h = int(np.fromfile(f, np.int32, count=1))

        if short:
            flow = np.fromfile(f, np.int16, count=h * w * 2)
            flow = flow.astype(np.float32)
        else:
            flow = np.fromfile(f, np.float32, count=h * w * 2)
        flow = flow.reshape((h, w, 2))

        flow = torch.from_numpy(flow)
        flow = flow.permute(2, 0, 1)
        transform = T.ToPILImage()
        flow = transform(flow)

    return flow


def flowToMap(F_mag, F_dir):
    sz = F_mag.shape
    flow_color = np.zeros((sz[0], sz[1], 3), dtype=float)
    # flow_color[:, :, 0] = (F_dir + np.pi) / (2 * np.pi)
    flow_color[:, :, 0] = (F_dir + np.pi) / (2 * np.pi)
    f_dir = (F_dir + np.pi) / (2 * np.pi)
    flow_color[:, :, 1] = F_mag / 255  # F_mag.max()
    flow_color[:, :, 2] = 1
    flow_color = matplotlib.colors.hsv_to_rgb(flow_color) * 255
    return flow_color


def flowToColor(flow):
    F_dx = flow[:, :, 1].copy().astype(float)
    F_dy = flow[:, :, 0].copy().astype(float)
    F_mag = np.sqrt(np.power(F_dx, 2) + np.power(F_dy, 2))
    F_dir = np.arctan2(F_dy, F_dx)
    flow_color = flowToMap(F_mag, F_dir)
    return flow_color.astype(np.uint8)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.avg = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.cnt = self.cnt + n
        self.total = self.total + val * n
        self.avg = self.total / self.cnt


def render(mask, rho, flow, ref):
    """
    actual reconstruction function used in loss.py
    :param mask: size: [N, 2, H, W]; without softmax
    :param rho: size: [N, 1, H, W]; without sigmoid
    :param flow: size: [N, 2, H, W]; without tanh
    :param ref: size: [N, 3, H, W]; range in [0, 1]
    :return: reconstruction image Tensor, size: [N, 3, H, W]; range in [0, 1]
    """
    n, c, h, w = ref.size()
    # mask_normed = torch.softmax(mask, dim=1)
    # rho_normed = torch.sigmoid(rho)
    flow_normed = torch.tanh(flow)
    mask_normed = torch.argmax(mask, dim=1, keepdim=True).to(dtype=torch.float32)
    grid_x = np.tile(np.linspace(0, w - 1, w), (h, 1)).astype(float)
    grid_y = np.tile(np.linspace(0, h - 1, h), (w, 1)).T.astype(float)
    # flow_normed = flow.permute(0, 2, 3, 1)
    flow_normed = flow_normed.permute(0, 2, 3, 1)
    preds = []
    for i in range(n):
        ref = ref.permute(0, 2, 3, 1)
        warped = warpImage(ref[i].detach().numpy(), flow_normed[i].detach().numpy(), grid_x, grid_y)
        warped = warped.permute(2, 0, 1)
        ref = ref.permute(0, 3, 1, 2)
        pred = (1 - mask_normed[i]) * ref[i] + mask_normed[i] * rho[i] * warped
        # pred = (1 - mask_normed[i]) * ref[i] + mask_normed[i] * rho_normed[i] * warped
        pred = pred.unsqueeze(dim=0)
        preds.append(pred)
    pred_img = torch.cat(preds)
    return pred_img


def warpImage(ref, flow, grid_x, grid_y):
    h, w = grid_x.shape
    ref *= w
    flow *= w
    flow_x = np.clip(flow[:, :, 1] + grid_x, 0, w - 1)
    flow_y = np.clip(flow[:, :, 0] + grid_y, 0, h - 1)
    flow_x, flow_y = cv2.convertMaps(flow_x.astype(np.float32), flow_y.astype(np.float32), cv2.CV_32FC2)
    warped_img = cv2.remap(ref, flow_x, flow_y, cv2.INTER_LINEAR)
    warped_img = torch.from_numpy(warped_img)
    return warped_img


def init_weights(layer):
    if type(layer) == Conv2d:
        torch.nn.init.kaiming_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)
