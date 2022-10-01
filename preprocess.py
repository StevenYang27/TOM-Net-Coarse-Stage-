import torch
import torchvision.transforms as T

TRAIN_SIMPLE_2k = {
    'number': 2000,
    'mean': [0.48, 0.46, 0.43],
    'std': [0.27, 0.26, 0.27],
    'flow_mean': [0.0562, 0.0483],
    'flow_std': [0.2068, 0.1889]
}

TRAIN_ALL_178k = {
    'number': 178000,
    'mean': [0.47, 0.45, 0.42],
    'std': [0.27, 0.27, 0.28],
    'flow_mean': [0.0631, 0.0642],
    'flow_std': [0.2118, 0.2126]
}

DATASET_STAT = {'simple': TRAIN_SIMPLE_2k, 'complete': TRAIN_ALL_178k}


class TOMTransform(object):
    def __init__(self, train, dataset, p=0.5):
        self.train = train
        self.dataset = dataset
        self.p = p
        self.general_transform_list = [
            T.Resize(512),
            T.ToTensor()
        ]
        self.data_transform_list = [
            T.ToPILImage(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Lambda(lambda x: x + torch.randn_like(x)),
            T.Normalize(mean=self.dataset['mean'],
                        std=self.dataset['std']),
        ]
        self.flow_transform_list = [
            T.Normalize(mean=self.dataset['flow_mean'],
                        std=self.dataset['flow_std'])

        ]
        self.general_transform = T.Compose(self.general_transform_list)
        self.data_transform = T.Compose(self.data_transform_list)
        self.flow_transform = T.Compose(self.flow_transform_list)

    def __call__(self, inputs, rand, is_data=False):
        out = self.general_transform(inputs)
        c, h, w = out.shape
        if self.train:
            if is_data:  # image data
                out = self.data_transform(out)
            elif c == 2:  # flow image
                out = self.flow_transform(out)
            if self.p > rand:
                out.flip(1)  # vertical flip
            if self.p > rand:
                out.flip(2)  # horizontal flip

        return out
