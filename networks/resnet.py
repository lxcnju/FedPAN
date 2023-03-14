import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# from networks.pe_func import sin_pe_func
import torch
import numpy as np


def sin_pe_func(pe_op, pe_t, pe_alpha, pe_ratio, n_hidden):
    # T: 0.5, 1.0, 2.0, 4.0, 8.0, 32.0
    indx = torch.arange(n_hidden) / n_hidden
    T = pe_t
    mask = torch.sin(2.0 * np.pi * indx * T)

    if pe_op == "add":
        mask = pe_alpha * mask
    elif pe_op == "mul":
        mask = pe_alpha * mask + 1.0
    else:
        pass

    # mask ratio
    n = int(pe_ratio * n_hidden)

    if pe_op == "add":
        mask[n:] = 0.0
    elif pe_op == "mul":
        mask[n:] = 1.0
    else:
        pass

    mask = mask.reshape((1, -1))

    return mask


class PEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, pe_way, pe_t, pe_alpha, pe_op, pe_ratio,
            in_planes, planes, stride=1):
        super().__init__()
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.pe1 = sin_pe_func(
            pe_op=self.pe_op,
            pe_t=self.pe_t,
            pe_alpha=self.pe_alpha,
            pe_ratio=self.pe_ratio,
            n_hidden=planes
        )
        self.pe1 = self.pe1.reshape((1, -1, 1, 1))

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.pe2 = sin_pe_func(
            pe_op=self.pe_op,
            pe_t=self.pe_t,
            pe_alpha=self.pe_alpha,
            pe_ratio=self.pe_ratio,
            n_hidden=planes
        )
        self.pe2 = self.pe2.reshape((1, -1, 1, 1))

        self.sc = False
        self.sc_mat = torch.diag(torch.ones(planes))

        if stride != 1 or in_planes != planes:
            self.sc = True

            self.sc_conv = nn.Conv2d(
                in_planes, planes,
                kernel_size=1, stride=stride, bias=False
            )
            self.sc_bn = nn.BatchNorm2d(planes)

            self.sc_pe = sin_pe_func(
                pe_op=self.pe_op,
                pe_t=self.pe_t,
                pe_alpha=self.pe_alpha,
                pe_ratio=self.pe_ratio,
                n_hidden=self.expansion * planes
            )
            self.sc_pe = self.sc_pe.reshape((1, -1, 1, 1))

    def forward(self, x):
        self.pe1 = self.pe1.to(x.device)
        self.pe2 = self.pe2.to(x.device)

        if self.sc is True:
            self.sc_pe = self.sc_pe.to(x.device)

        if self.pe_op == "add":
            out = F.relu(self.bn1(self.conv1(x)) + self.pe1)
            out = self.bn2(self.conv2(out)) + self.pe2

            if self.sc is True:
                out += self.sc_bn(self.sc_conv(x)) + self.sc_pe
            else:
                B, C, W, H = x.shape
                sc_x = x.transpose(0, 1).reshape(C, -1)
                self.sc_mat = self.sc_mat.to(x.device)
                sc_x = torch.mm(self.sc_mat, sc_x)
                sc_x = sc_x.reshape((C, B, W, H)).transpose(0, 1)
                out += sc_x

            out = F.relu(out)
        elif self.pe_op == "mul":
            out = F.relu(self.bn1(self.conv1(x)) * self.pe1)
            out = self.bn2(self.conv2(out)) * self.pe2

            if self.sc is True:
                out += self.sc_bn(self.sc_conv(x)) * self.sc_pe
            else:
                B, C, W, H = x.shape
                sc_x = x.transpose(0, 1).reshape(C, -1)
                self.sc_mat = self.sc_mat.to(x.device)
                sc_x = torch.mm(self.sc_mat, sc_x)
                sc_x = sc_x.reshape((C, B, W, H)).transpose(0, 1)
                out += sc_x

            out = F.relu(out)

        return out

    def generate_sf_inds(self, size, sf_ratio, sf_prob):
        nsf = int(size * sf_ratio)

        inds = list(range(size))
        for i in range(size):
            j = random.choice(range(i, min(size, i + nsf + 1)))

            if random.random() <= sf_prob:
                temp = inds[i]
                inds[i] = inds[j]
                inds[j] = temp

        inds = torch.LongTensor(inds)
        return inds

    def shuffle(self, sf_ratio, sf_prob, inds0=None, last_sf=False):
        inds1 = self.generate_sf_inds(
            size=self.planes, sf_ratio=sf_ratio, sf_prob=sf_prob
        )
        inds2 = self.generate_sf_inds(
            size=self.planes, sf_ratio=sf_ratio, sf_prob=sf_prob
        )

        if inds0 is not None:
            self.conv1.weight.data = self.conv1.weight.data[:, inds0]

            if self.sc is True:
                self.sc_conv.weight.data = self.sc_conv.weight.data[:, inds0]
            else:
                self.sc_mat = self.sc_mat[:, inds0]

        self.conv1.weight.data = self.conv1.weight.data[inds1]
        self.bn1.weight.data = self.bn1.weight.data[inds1]
        self.bn1.bias.data = self.bn1.bias.data[inds1]
        self.bn1.running_mean.data = self.bn1.running_mean.data[inds1]
        self.bn1.running_var.data = self.bn1.running_var.data[inds1]

        self.conv2.weight.data = self.conv2.weight.data[:, inds1]

        if last_sf is True:
            self.conv2.weight.data = self.conv2.weight.data[inds2]
            self.bn2.weight.data = self.bn2.weight.data[inds2]
            self.bn2.bias.data = self.bn2.bias.data[inds2]
            self.bn2.running_mean.data = self.bn2.running_mean.data[inds2]
            self.bn2.running_var.data = self.bn2.running_var.data[inds2]

            if self.sc is True:
                self.sc_conv.weight.data = self.sc_conv.weight.data[inds2]
                self.sc_bn.weight.data = self.sc_bn.weight.data[inds2]
                self.sc_bn.bias.data = self.sc_bn.bias.data[inds2]
                self.sc_bn.running_mean.data = self.sc_bn.running_mean.data[
                    inds2
                ]
                self.sc_bn.running_var.data = self.sc_bn.running_var.data[
                    inds2
                ]
            else:
                self.sc_mat = self.sc_mat[inds2]

        return inds2


class PEBottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, pe_way, pe_t, pe_alpha, pe_op, pe_ratio,
            in_planes, planes, stride=1):
        super().__init__()
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.pe1 = sin_pe_func(
            pe_op=self.pe_op,
            pe_t=self.pe_t,
            pe_alpha=self.pe_alpha,
            pe_ratio=self.pe_ratio,
            n_hidden=planes
        )
        self.pe1 = self.pe1.reshape((1, -1, 1, 1))

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.pe2 = sin_pe_func(
            pe_op=self.pe_op,
            pe_t=self.pe_t,
            pe_alpha=self.pe_alpha,
            pe_ratio=self.pe_ratio,
            n_hidden=planes
        )
        self.pe2 = self.pe2.reshape((1, -1, 1, 1))

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.pe3 = sin_pe_func(
            pe_op=self.pe_op,
            pe_t=self.pe_t,
            pe_alpha=self.pe_alpha,
            pe_ratio=self.pe_ratio,
            n_hidden=self.expansion * planes
        )
        self.pe3 = self.pe3.reshape((1, -1, 1, 1))

        self.sc = False
        self.sc_mat = torch.diag(torch.ones(self.expansion * planes))

        if stride != 1 or in_planes != self.expansion * planes:
            self.sc = True

            self.sc_conv = nn.Conv2d(
                in_planes, self.expansion * planes,
                kernel_size=1, stride=stride, bias=False
            )
            self.sc_bn = nn.BatchNorm2d(self.expansion * planes)

            self.sc_pe = sin_pe_func(
                pe_op=self.pe_op,
                pe_t=self.pe_t,
                pe_alpha=self.pe_alpha,
                pe_ratio=self.pe_ratio,
                n_hidden=self.expansion * planes
            )
            self.sc_pe = self.sc_pe.reshape((1, -1, 1, 1))

    def generate_sf_inds(self, size, sf_ratio, sf_prob):
        nsf = int(size * sf_ratio)

        inds = list(range(size))
        for i in range(size):
            j = random.choice(range(i, min(size, i + nsf + 1)))

            if random.random() <= sf_prob:
                temp = inds[i]
                inds[i] = inds[j]
                inds[j] = temp

        inds = torch.LongTensor(inds)
        return inds

    def forward(self, x):
        self.pe1 = self.pe1.to(x.device)
        self.pe2 = self.pe2.to(x.device)
        self.pe3 = self.pe3.to(x.device)

        if self.sc is True:
            self.sc_pe = self.sc_pe.to(x.device)

        if self.pe_op == "add":
            out = F.relu(self.bn1(self.conv1(x)) + self.pe1)
            out = F.relu(self.bn2(self.conv2(out)) + self.pe2)
            out = self.bn3(self.conv3(out)) + self.pe3

            if self.sc is True:
                out += self.sc_bn(self.sc_conv(x)) + self.sc_pe
            else:
                B, C, W, H = x.shape
                sc_x = x.transpose(0, 1).reshape(C, -1)
                self.sc_mat = self.sc_mat.to(x.device)
                sc_x = torch.mm(self.sc_mat, sc_x)
                sc_x = sc_x.reshape((C, B, W, H)).transpose(0, 1)
                out += sc_x

            out = F.relu(out)
        elif self.pe_op == "mul":
            out = F.relu(self.bn1(self.conv1(x)) * self.pe1)
            out = F.relu(self.bn2(self.conv2(out)) * self.pe2)
            out = self.bn3(self.conv3(out)) * self.pe3

            if self.sc is True:
                out += self.sc_bn(self.sc_conv(x)) * self.sc_pe
            else:
                B, C, W, H = x.shape
                sc_x = x.transpose(0, 1).reshape(C, -1)
                self.sc_mat = self.sc_mat.to(x.device)
                sc_x = torch.mm(self.sc_mat, sc_x)
                sc_x = sc_x.reshape((C, B, W, H)).transpose(0, 1)
                out += sc_x

            out = F.relu(out)
        return out

    def shuffle(self, sf_ratio, sf_prob, inds0=None, last_sf=False):
        inds1 = self.generate_sf_inds(
            size=self.planes, sf_ratio=sf_ratio, sf_prob=sf_prob
        )
        inds2 = self.generate_sf_inds(
            size=self.planes, sf_ratio=sf_ratio, sf_prob=sf_prob
        )
        inds3 = self.generate_sf_inds(
            size=self.expansion * self.planes,
            sf_ratio=sf_ratio, sf_prob=sf_prob
        )

        if inds0 is not None:
            self.conv1.weight.data = self.conv1.weight.data[:, inds0]

            if self.sc is True:
                self.sc_conv.weight.data = self.sc_conv.weight.data[:, inds0]
            else:
                self.sc_mat = self.sc_mat[:, inds0]

        self.conv1.weight.data = self.conv1.weight.data[inds1]
        self.bn1.weight.data = self.bn1.weight.data[inds1]
        self.bn1.bias.data = self.bn1.bias.data[inds1]
        self.bn1.running_mean.data = self.bn1.running_mean.data[inds1]
        self.bn1.running_var.data = self.bn1.running_var.data[inds1]

        self.conv2.weight.data = self.conv2.weight.data[:, inds1]

        self.conv2.weight.data = self.conv2.weight.data[inds2]
        self.bn2.weight.data = self.bn2.weight.data[inds2]
        self.bn2.bias.data = self.bn2.bias.data[inds2]
        self.bn2.running_mean.data = self.bn2.running_mean.data[inds2]
        self.bn2.running_var.data = self.bn2.running_var.data[inds2]

        self.conv3.weight.data = self.conv3.weight.data[:, inds2]

        if last_sf is True:
            self.conv3.weight.data = self.conv3.weight.data[inds3]
            self.bn3.weight.data = self.bn3.weight.data[inds3]
            self.bn3.bias.data = self.bn3.bias.data[inds3]
            self.bn3.running_mean.data = self.bn3.running_mean.data[inds3]
            self.bn3.running_var.data = self.bn3.running_var.data[inds3]

            if self.sc is True:
                self.sc_conv.weight.data = self.sc_conv.weight.data[inds3]
                self.sc_bn.weight.data = self.sc_bn.weight.data[inds3]
                self.sc_bn.bias.data = self.sc_bn.bias.data[inds3]
                self.sc_bn.running_mean.data = self.sc_bn.running_mean.data[
                    inds3
                ]
                self.sc_bn.running_var.data = self.sc_bn.running_var.data[
                    inds3
                ]
            else:
                self.sc_mat = self.sc_mat[inds3]

        return inds3


class CifarPEResNet(nn.Module):
    def __init__(
            self, pe_way, pe_t, pe_alpha, pe_op, pe_ratio,
            n_layer=18, n_classes=10):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio

        self.pe = sin_pe_func(
            pe_op=self.pe_op,
            pe_t=1.0,
            pe_alpha=self.pe_alpha,
            pe_ratio=0.0,
            n_hidden=64
        )
        self.pe = self.pe.reshape((1, -1, 1, 1))

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        assert ((n_layer - 2) % 6 == 0), "SmallResNet depth is 6n+2"
        n = int((n_layer - 2) / 6)

        self.cfg = (PEBasicBlock, (n, n, n))
        self.in_planes = 64

        self.layers1 = self._make_layer(
            block=self.cfg[0], planes=64, stride=1, num_blocks=self.cfg[1][0],
        )
        self.layers2 = self._make_layer(
            block=self.cfg[0], planes=128, stride=2, num_blocks=self.cfg[1][1],
        )
        self.layers3 = self._make_layer(
            block=self.cfg[0], planes=256, stride=2, num_blocks=self.cfg[1][2],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            256 * self.cfg[0].expansion, n_classes
        )

    def _make_layer(self, block, planes, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()

        for stride in strides:
            layers.append(block(
                pe_way=self.pe_way,
                pe_op=self.pe_op,
                pe_t=self.pe_t,
                pe_alpha=self.pe_alpha,
                pe_ratio=self.pe_ratio,
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
            ))
            self.in_planes = block.expansion * planes
        return layers

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        if self.pe_op == "add":
            out = F.relu(self.bn1(self.conv1(x)) + self.pe)
        elif self.pe_op == "mul":
            out = F.relu(self.bn1(self.conv1(x)) * self.pe)

        for layer in self.layers1:
            out = layer(out)

        for layer in self.layers2:
            out = layer(out)

        for layer in self.layers3:
            out = layer(out)

        out = self.avgpool(out)
        hs = out.view(out.size(0), -1)
        out = self.fc(hs)
        return hs, out

    def shuffle(self, sf_ratio=0.0, sf_prob=0.0, reshuffle=True):
        inds0 = None
        for layer in self.layers1:
            inds0 = layer.shuffle(
                sf_ratio=sf_ratio, sf_prob=sf_prob, inds0=inds0, last_sf=True
            )

        for layer in self.layers2:
            inds0 = layer.shuffle(
                sf_ratio=sf_ratio, sf_prob=sf_prob, inds0=inds0, last_sf=True
            )

        for layer in self.layers3:
            inds0 = layer.shuffle(
                sf_ratio=sf_ratio, sf_prob=sf_prob, inds0=inds0, last_sf=True
            )

        self.fc.weight.data = self.fc.weight.data[:, inds0]


class SelfPEResNet(nn.Module):
    def __init__(
            self, pe_way, pe_t, pe_alpha, pe_op, pe_ratio,
            first_conv="7x7-nopool", n_layer=18, n_classes=10):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes
        self.first_conv = first_conv
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio

        if first_conv == "7x7-pool":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7,
                stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(
                3, stride=2, padding=1
            )
        elif first_conv == "7x7-nopool":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7,
                stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
        elif first_conv == "3x3-1":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3,
                stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
        elif first_conv == "3x3-2":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3,
                stride=2, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
        else:
            raise ValueError("No such first conv: {}".format(first_conv))

        self.pe = sin_pe_func(
            pe_op=self.pe_op,
            pe_t=1.0,
            pe_alpha=self.pe_alpha,
            pe_ratio=0.0,
            n_hidden=64
        )
        self.pe = self.pe.reshape((1, -1, 1, 1))

        if n_layer == 18:
            self.cfg = (PEBasicBlock, (2, 2, 2, 2))
        elif n_layer == 34:
            self.cfg = (PEBasicBlock, (3, 4, 6, 3))
        elif n_layer == 50:
            self.cfg = (PEBottleneck, (3, 4, 6, 3))
        elif n_layer == 101:
            self.cfg = (PEBottleneck, (3, 4, 23, 3))
        elif n_layer == 152:
            self.cfg = (PEBottleneck, (3, 8, 36, 3))
        else:
            raise ValueError("No such n_layer: {}".format(n_layer))

        self.in_planes = 64

        self.layers1 = self._make_layer(
            block=self.cfg[0], planes=64, stride=1, num_blocks=self.cfg[1][0],
        )
        self.layers2 = self._make_layer(
            block=self.cfg[0], planes=128, stride=2, num_blocks=self.cfg[1][1],
        )
        self.layers3 = self._make_layer(
            block=self.cfg[0], planes=256, stride=2, num_blocks=self.cfg[1][2],
        )
        self.layers4 = self._make_layer(
            block=self.cfg[0], planes=512, stride=2, num_blocks=self.cfg[1][3],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            512 * self.cfg[0].expansion, n_classes
        )

    def _make_layer(self, block, planes, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()

        for stride in strides:
            layers.append(block(
                pe_way=self.pe_way,
                pe_op=self.pe_op,
                pe_t=self.pe_t,
                pe_alpha=self.pe_alpha,
                pe_ratio=self.pe_ratio,
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
            ))
            self.in_planes = block.expansion * planes
        return layers

    def forward(self, x):
        self.pe = self.pe.to(x.device)

        if self.pe_op == "add":
            out = F.relu(self.bn1(self.conv1(x)) + self.pe)
        elif self.pe_op == "mul":
            out = F.relu(self.bn1(self.conv1(x)) * self.pe)

        if self.first_conv == "7x7-pool":
            out = self.pool1(out)

        for layer in self.layers1:
            out = layer(out)

        for layer in self.layers2:
            out = layer(out)

        for layer in self.layers3:
            out = layer(out)

        for layer in self.layers4:
            out = layer(out)

        out = self.avgpool(out)
        hs = out.view(out.size(0), -1)
        out = self.fc(hs)
        return hs, out

    def shuffle(self, sf_ratio=0.0, sf_prob=0.0, reshuffle=True):
        inds0 = None
        for layer in self.layers1:
            inds0 = layer.shuffle(
                sf_ratio=sf_ratio, sf_prob=sf_prob, inds0=inds0, last_sf=True
            )

        for layer in self.layers2:
            inds0 = layer.shuffle(
                sf_ratio=sf_ratio, sf_prob=sf_prob, inds0=inds0, last_sf=True
            )
        for layer in self.layers3:
            inds0 = layer.shuffle(
                sf_ratio=sf_ratio, sf_prob=sf_prob, inds0=inds0, last_sf=True
            )

        for layer in self.layers4:
            inds0 = layer.shuffle(
                sf_ratio=sf_ratio, sf_prob=sf_prob, inds0=inds0, last_sf=True
            )

        self.fc.weight.data = self.fc.weight.data[:, inds0]


if __name__ == "__main__":
    """
    for n_layer in [18, 34, 50, 101]:
        model = SelfResNet(n_layer=n_layer, n_classes=10)
        n_params = sum([
            param.numel() for param in model.parameters()
        ])
        print("Total number of parameters : {}".format(n_params))

        xs = torch.randn(1, 3, 224, 224)
        hs = model(xs)
        print(hs.shape)
    """

    """
    p1 = 32
    p2 = 32
    p3 = 32
    for pe_alpha in [0.0, 0.01, 0.05, 0.1, 0.2]:
        model1 = PEBasicBlock(
            pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
            pe_ratio=1.0, pe_t=1.0,
            in_planes=32, planes=p1,
            stride=1,
        )
        model1.eval()

        model2 = PEBasicBlock(
            pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
            pe_ratio=1.0, pe_t=1.0,
            in_planes=p1, planes=p2,
            stride=1,
        )
        model2.eval()

        model3 = PEBasicBlock(
            pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
            pe_ratio=1.0, pe_t=1.0,
            in_planes=p2, planes=p3,
            stride=1,
        )
        model3.eval()

        xs = torch.randn(4, 32, 96, 96)
        out1 = model3(model2(model1(xs)))
        out2 = model3(model2(model1(xs)))

        inds0 = model1.shuffle(sf_ratio=1.0, inds0=None, last_sf=True)
        inds0 = model2.shuffle(sf_ratio=1.0, inds0=inds0, last_sf=True)
        model3.shuffle(sf_ratio=1.0, inds0=inds0, last_sf=False)
        out3 = model3(model2(model1(xs)))

        print(torch.abs(out1 - out2).mean())
        print(torch.abs(out1 - out3).mean())

        out1 = model3(model2(model1(xs)))
        out2 = model3(model2(model1(xs)))

        inds0 = model1.shuffle(sf_ratio=1.0, inds0=None, last_sf=True)
        inds0 = model2.shuffle(sf_ratio=1.0, inds0=inds0, last_sf=True)
        model3.shuffle(sf_ratio=1.0, inds0=inds0, last_sf=False)
        out3 = model3(model2(model1(xs)))

        print(torch.abs(out1 - out2).mean())
        print(torch.abs(out1 - out3).mean())
    """

    for n_layer in [20, 32, 44, 56, 110]:
        for pe_alpha in [0.0, 0.01, 0.05, 0.1]:
            model = CifarPEResNet(
                pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
                pe_ratio=1.0, pe_t=8.0, n_layer=n_layer, n_classes=10
            )
            model.eval()

            xs = torch.randn(20, 3, 32, 32)
            out1 = model(xs)[1]
            out2 = model(xs)[1]

            model.shuffle(sf_ratio=1.0, sf_prob=1.0)
            out3 = model(xs)[1]

            print(torch.abs(out1 - out2).mean().item())
            print(torch.abs(out1 - out3).mean().item())

            xs = torch.randn(20, 3, 32, 32)
            out1 = model(xs)[1]
            out2 = model(xs)[1]

            model.shuffle(sf_ratio=1.0, sf_prob=1.0)
            out3 = model(xs)[1]

            print(torch.abs(out1 - out2).mean().item())
            print(torch.abs(out1 - out3).mean().item())

    """
    p1 = 32
    p2 = 32
    p3 = 32
    for pe_alpha in [0.0, 0.01, 0.05, 0.1, 0.2]:
        model1 = PEBottleneck(
            pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
            pe_ratio=1.0, pe_t=1.0,
            in_planes=32, planes=p1,
            stride=1,
        )
        model1.eval()

        model2 = PEBottleneck(
            pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
            pe_ratio=1.0, pe_t=1.0,
            in_planes=4 * p1, planes=p2,
            stride=1,
        )
        model2.eval()

        model3 = PEBottleneck(
            pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
            pe_ratio=1.0, pe_t=1.0,
            in_planes=4 * p2, planes=p3,
            stride=1,
        )
        model3.eval()

        xs = torch.randn(4, 32, 96, 96)
        out1 = model3(model2(model1(xs)))
        out2 = model3(model2(model1(xs)))

        inds0 = model1.shuffle(sf_ratio=1.0, inds0=None, last_sf=True)
        inds0 = model2.shuffle(sf_ratio=1.0, inds0=inds0, last_sf=True)
        model3.shuffle(sf_ratio=1.0, inds0=inds0, last_sf=False)
        out3 = model3(model2(model1(xs)))

        print(torch.abs(out1 - out2).mean())
        print(torch.abs(out1 - out3).mean())

        out1 = model3(model2(model1(xs)))
        out2 = model3(model2(model1(xs)))

        inds0 = model1.shuffle(sf_ratio=1.0, inds0=None, last_sf=True)
        inds0 = model2.shuffle(sf_ratio=1.0, inds0=inds0, last_sf=True)
        model3.shuffle(sf_ratio=1.0, inds0=inds0, last_sf=False)
        out3 = model3(model2(model1(xs)))

        print(torch.abs(out1 - out2).mean())
        print(torch.abs(out1 - out3).mean())
    """

    for n_layer in [18, 34, 50, 101]:
        for pe_alpha in [0.0, 0.01, 0.05, 0.1]:
            model = SelfPEResNet(
                pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
                pe_ratio=1.0, pe_t=1.0, n_layer=n_layer, n_classes=10
            )
            model.eval()

            xs = torch.randn(20, 3, 32, 32)
            out1 = model(xs)
            out2 = model(xs)

            model.shuffle(sf_ratio=1.0, sf_prob=1.0)
            out3 = model(xs)

            print(torch.abs(out1 - out2).mean())
            print(torch.abs(out1 - out3).mean())

            xs = torch.randn(20, 3, 32, 32)
            out1 = model(xs)
            out2 = model(xs)

            model.shuffle(sf_ratio=1.0, sf_prob=1.0)
            out3 = model(xs)

            print(torch.abs(out1 - out2).mean())
            print(torch.abs(out1 - out3).mean())

