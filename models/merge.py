import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.vertical_deploy = False
        self.horizontal_deploy = False
        self.block0 = torch.nn.Sequential(nn.Conv2d(in_channels=3, out_channels=127, kernel_size=3, stride=1,
                                                    padding=0, dilation=1, groups=1, bias=True),
                                          nn.Conv2d(in_channels=127, out_channels=127, kernel_size=1, stride=1,
                                                    padding=0, dilation=1, groups=1, bias=True))

    def switch_to_deploy(self):

        kernel = self.block0[2].weight.data
        bias = self.block0[2].bias.data

        self.conv1.weight.data = torch.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2),
                                              self.block0[1].weight.data)
        self.conv1.bias.data = bias + (self.conv1.bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.conv1 = torch.nn.Sequential(*[self.conv1])
        self.__delattr__('conv2')
        self.vertical_deploy = True

    def forward(self, x):
        if self.vertical_deploy:
            x = self.conv1(x)
            return x
        else:
            x = self.block0(x)
            return x


if __name__ == '__main__':
    input = torch.rand([1, 3, 127, 127])
    net = Block()
    out1 = net(input)
    net.switch_to_deploy()
    out2 = net(input)
    print(((out1 - out2) ** 2).sum())

