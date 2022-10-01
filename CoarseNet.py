import numpy as np
import torch
from torch.nn import (BatchNorm2d, Conv2d, LeakyReLU, Module, ModuleList,
                      Sequential, UpsamplingNearest2d)


def ConvBNRelu(in_channels, out_channels, stride, up=False):
    layers = Sequential(Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               stride=stride,
                               kernel_size=3,
                               padding=1),
                        BatchNorm2d(num_features=out_channels),
                        LeakyReLU(inplace=False))
    if up:
        layers = Sequential(layers, UpsamplingNearest2d(scale_factor=2))

    return layers


class Encoder(Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer1 = ConvBNRelu(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 stride=2)
        self.layer2 = ConvBNRelu(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 stride=1)
        self.block = Sequential(self.layer1, self.layer2)

    def forward(self, inputs):
        output = self.block(inputs)
        return output


class Decoder(Module):
    def __init__(self, in_channels, out_channels, include_final_output):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_final_output = include_final_output

        self.layer = ConvBNRelu(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                stride=1,
                                up=True)
        self.up = UpsamplingNearest2d(scale_factor=2)
        self.block = ModuleList()
        for i in range(1, 4):
            self.block.append(self.layer)
        if self.include_final_output:
            self.block.append(ConvBNRelu(in_channels=self.in_channels,
                                         out_channels=2,
                                         stride=1))  # mask
            self.block.append(ConvBNRelu(in_channels=self.in_channels,
                                         out_channels=1,
                                         stride=1))  # attenuation
            self.block.append(ConvBNRelu(in_channels=self.in_channels,
                                         out_channels=2,
                                         stride=1))  # flow

    def forward(self, inputs):
        output = [self.block[i](inputs) for i in range(len(self.block))]
        if self.include_final_output:
            final_output = output[3:]
            inter_output = output[:3]
            temp = [output[i].clone() for i in range(3, 6)]
            for i in range(3):
                inter_output.append(self.up(temp[i]))
            return inter_output, final_output
        else:
            return output


class CoarseNet(Module):
    def __init__(self):
        super(CoarseNet, self).__init__()
        self.scales = [1, 2, 3, 4]
        self.up = UpsamplingNearest2d(scale_factor=2)
        # Encoder
        self.down1 = Sequential(ConvBNRelu(in_channels=3, out_channels=16, stride=1),
                                ConvBNRelu(in_channels=16, out_channels=16, stride=1))  # 1/1
        self.down2 = Encoder(in_channels=16, out_channels=16)  # 1/2
        self.down3 = Encoder(in_channels=16, out_channels=32)  # 1/4
        self.down4 = Encoder(in_channels=32, out_channels=64)  # 1/8
        self.down5 = Encoder(in_channels=64, out_channels=128)  # 1/16
        self.down6 = Encoder(in_channels=128, out_channels=256)  # 1/32
        self.down7 = Encoder(in_channels=256, out_channels=256)  # 1/64
        # Decoder
        self.up7 = Decoder(in_channels=256, out_channels=256, include_final_output=False)  # 1/32
        self.up6 = Decoder(in_channels=4 * 256, out_channels=128, include_final_output=False)  # 1/16
        self.up5 = Decoder(in_channels=4 * 128, out_channels=64, include_final_output=False)  # 1/8
        self.up4 = Decoder(in_channels=4 * 64, out_channels=32, include_final_output=True)  # 1/4
        self.up3 = Decoder(in_channels=4 * 32 + 5, out_channels=16, include_final_output=True)  # 1/2
        self.up2 = Decoder(in_channels=4 * 16 + 5, out_channels=16, include_final_output=True)  # 1/1
        self.up1 = ModuleList([ConvBNRelu(in_channels=4 * 16 + 5, out_channels=2, stride=1),  # mask
                               ConvBNRelu(in_channels=4 * 16 + 5, out_channels=1, stride=1),  # rho
                               ConvBNRelu(in_channels=4 * 16 + 5, out_channels=2, stride=1)])  # flow

    def forward(self, inputs):
        # Encoder
        encoder1_out = self.down1(inputs)
        encoder2_out = self.down2(encoder1_out)
        encoder3_out = self.down3(encoder2_out)
        encoder4_out = self.down4(encoder3_out)
        encoder5_out = self.down5(encoder4_out)
        encoder6_out = self.down6(encoder5_out)
        encoder7_out = self.down7(encoder6_out)

        # Decoder
        decoder7_outs = self.up7(encoder7_out)
        decoder6_ins = torch.cat((*decoder7_outs, encoder6_out), dim=1)

        decoder6_outs = self.up6(decoder6_ins)
        decoder5_ins = torch.cat((*decoder6_outs, encoder5_out), dim=1)

        decoder5_outs = self.up5(decoder5_ins)
        decoder4_ins = torch.cat((*decoder5_outs, encoder4_out), dim=1)

        decoder4_outs, final_out4 = self.up4(decoder4_ins)
        decoder3_ins = torch.cat((*decoder4_outs, encoder3_out), dim=1)

        decoder3_outs, final_out3 = self.up3(decoder3_ins)
        decoder2_ins = torch.cat((*decoder3_outs, encoder2_out), dim=1)

        decoder2_outs, final_out2 = self.up2(decoder2_ins)
        decoder1_ins = torch.cat((*decoder2_outs, encoder1_out), dim=1)

        final_out1 = [self.up1[i](decoder1_ins) for i in range(len(self.up1))]

        final_out = [final_out1, final_out2, final_out3, final_out4]
        final_out = dict(zip(self.scales, final_out))

        return final_out


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoarseNet()
    model.to(device)
    model.train()
    x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
    x_variable = torch.autograd.Variable(torch.from_numpy(x_data))
    print(f'Input size: \n{x_variable.size()}')
    pred = model(x_variable)
    print(f'Output size: \n'
          f'Mask size: {pred[1][0].size()}\n'
          f'Attenuation size:{pred[1][1].size()}\n'
          f'Flow size: {pred[1][2].size()}')


if __name__ == '__main__':
    main()
