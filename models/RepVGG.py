from __future__ import absolute_import

import torch.nn as nn

from typing import Optional, List, Tuple, Union, Any

import copy
import torch
import torch.nn.functional as F
from torchstat import stat
from torchsummary import summary
from timm.models.layers import weight_init, DropPath
from timm.models.registry import register_model

__all__ = ['MobileOne', 'reparameterize_model']


class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=(self.act_num * 2 + 1) // 2, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        self.crop = int((self.kernel_size - 1) / 2)

        # 网络单一结构
        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=1,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.reparam_conv(x)

        # crop, for no padding reparam
        if (self.rbr_skip is not None or self.rbr_scale is not None) and self.kernel_size != 1:
            t = x[:, :, self.crop:-self.crop, self.crop:-self.crop].contiguous()
        else:
            t = x

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(t)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(t)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return out

    # 先设计网络的结构，在进行冲参数化的设计，和RepVGG差不多，应该问题很大
    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    # fuse bn tensor
    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    # 1*1 -> bn
    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=1,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileOne(nn.Module):
    """ MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(self, ) -> None:
        """ Construct MobileOne model.
        """
        super(MobileOne, self).__init__()
        self.act_learn = 1
        self.vertical_deploy = False
        self.block0 = nn.Sequential(MobileOneBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4),
                                    MobileOneBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4))
        self.act0 = activation(dim=64, act_num=3)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1 = nn.Sequential(MobileOneBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4),
                                    MobileOneBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4))
        self.act1 = activation(dim=128, act_num=3)

        self.block2 = nn.Sequential(MobileOneBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4),
                                    MobileOneBlock(in_channels=128, out_channels=128, kernel_size=1, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4))
        self.act2 = activation(dim=128, act_num=3)

        self.block3 = nn.Sequential(MobileOneBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4),
                                    MobileOneBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4))
        self.act3 = activation(dim=256, act_num=3)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = nn.Sequential(MobileOneBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4),
                                    MobileOneBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4))
        self.act4 = activation(dim=256, act_num=3)

        self.block5 = nn.Sequential(MobileOneBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4),
                                    MobileOneBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4))
        self.act5 = activation(dim=256, act_num=3)

        self.block6 = nn.Sequential(MobileOneBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4),
                                    MobileOneBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1,
                                                   padding=0, dilation=1, groups=1, inference_mode=False, use_se=False,
                                                   num_conv_branches=4))
        # self.act6 = activation(dim=256, act_num=3)

    def reparameter_model(self, block):
        kernel = block[1].reparam_conv.weight.data
        bias = block[1].reparam_conv.bias.data
        block[0].reparam_conv.weight.data = torch.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2),
                                                         block[0].reparam_conv.weight.data)
        block[0].reparam_conv.bias.data = bias + (
                block[0].reparam_conv.bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        block = torch.nn.Sequential(*[block[0]])
        self.vertical_deploy = True
        return block

    def test_reparam(self):
        for name, module in self.named_children():
            if 'block' in name:
                new_module = self.reparameter_model(module)
                setattr(self, name, new_module)
            if 'act' in name:
                module.switch_to_deploy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        if self.vertical_deploy:
            x = self.block0(x)
            x = self.act0(x)

            x = self.pool0(x)

            x = self.block1(x)
            x = self.act1(x)

            x = self.block2(x)
            x = self.act2(x)

            x = self.block3(x)
            x = self.act3(x)

            x = self.pool1(x)

            x = self.block4(x)
            x = self.act4(x)

            x = self.block5(x)
            x = self.act5(x)

            x = self.block6(x)

        else:
            x = self.block0[0](x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.block0[1](x)
            x = self.act0(x)

            x = self.pool0(x)

            x = self.block1[0](x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.block1[1](x)
            x = self.act1(x)

            x = self.block2[0](x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.block2[1](x)
            x = self.act2(x)

            x = self.block3[0](x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.block3[1](x)
            x = self.act3(x)

            x = self.pool1(x)

            x = self.block4[0](x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.block4[1](x)
            x = self.act4(x)

            x = self.block5[0](x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.block5[1](x)
            x = self.act5(x)

            x = self.block6[0](x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.block6[1](x)
            # x = self.act6(x)

        return x


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


def initial(net):
    for mod in net.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            nn.init.uniform_(mod.running_mean, 0, 0.1)
            nn.init.uniform_(mod.running_var, 0, 0.1)
            nn.init.uniform_(mod.weight, 0, 0.1)
            nn.init.uniform_(mod.bias, 0, 0.1)


"""
for deploy:
    net = reparameterize_model(net)
    net.test_reparam()
"""

from thop import profile

if __name__ == '__main__':
    net = MobileOne()
    initial(net)
    net.eval()

    input = torch.randn(1, 3, 127, 127)
    train_y = net(input)

    print("reparameterize before:\n", net)
    macs, params = profile(net, inputs=(input,))
    print(macs, params)

    net = reparameterize_model(net)
    net.test_reparam()

    print("\n\n\nreparameterize after:\n", net)

    deploy_y = net(input)
    # print(((train_y - deploy_y) ** 2).sum())
    macs, params = profile(net, inputs=(input,))
    print(macs, params)