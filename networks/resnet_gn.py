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


class PEBasicBlockGN(nn.Module):
    expansion = 1

    def __init__(
            self, pe_way, pe_t, pe_alpha, pe_op, pe_ratio,
            in_planes, planes, n_groups, stride=1):
        super().__init__()
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio
        self.in_planes = in_planes
        self.planes = planes
        self.n_groups = n_groups

        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(self.n_groups, planes)
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
        self.gn2 = nn.GroupNorm(self.n_groups, planes)
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
            self.sc_gn = nn.GroupNorm(self.n_groups, planes)

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
            out = F.relu(self.gn1(self.conv1(x)) + self.pe1)
            out = self.gn2(self.conv2(out)) + self.pe2

            if self.sc is True:
                out += self.sc_gn(self.sc_conv(x)) + self.sc_pe
            else:
                B, C, W, H = x.shape
                sc_x = x.transpose(0, 1).reshape(C, -1)
                self.sc_mat = self.sc_mat.to(x.device)
                sc_x = torch.mm(self.sc_mat, sc_x)
                sc_x = sc_x.reshape((C, B, W, H)).transpose(0, 1)
                out += sc_x

            out = F.relu(out)
        elif self.pe_op == "mul":
            out = F.relu(self.gn1(self.conv1(x)) * self.pe1)
            out = self.gn2(self.conv2(out)) * self.pe2

            if self.sc is True:
                out += self.sc_gn(self.sc_conv(x)) * self.sc_pe
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
        self.gn1.weight.data = self.gn1.weight.data[inds1]
        self.gn1.bias.data = self.gn1.bias.data[inds1]

        self.conv2.weight.data = self.conv2.weight.data[:, inds1]

        if last_sf is True:
            self.conv2.weight.data = self.conv2.weight.data[inds2]
            self.gn2.weight.data = self.gn2.weight.data[inds2]
            self.gn2.bias.data = self.gn2.bias.data[inds2]

            if self.sc is True:
                self.sc_conv.weight.data = self.sc_conv.weight.data[inds2]
                self.sc_gn.weight.data = self.sc_gn.weight.data[inds2]
                self.sc_gn.bias.data = self.sc_gn.bias.data[inds2]
            else:
                self.sc_mat = self.sc_mat[inds2]

        return inds2


class CifarPEResNetGN(nn.Module):
    def __init__(
            self, pe_way, pe_t, pe_alpha, pe_op, pe_ratio,
            n_layer, n_groups, n_classes=10):
        super().__init__()
        self.n_layer = n_layer
        self.n_groups = n_groups
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
        self.gn1 = nn.GroupNorm(self.n_groups, 64)

        assert ((n_layer - 2) % 6 == 0), "SmallResNet depth is 6n+2"
        n = int((n_layer - 2) / 6)

        self.cfg = (PEBasicBlockGN, (n, n, n))
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
                n_groups=self.n_groups
            ))
            self.in_planes = block.expansion * planes
        return layers

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        if self.pe_op == "add":
            out = F.relu(self.gn1(self.conv1(x)) + self.pe)
        elif self.pe_op == "mul":
            out = F.relu(self.gn1(self.conv1(x)) * self.pe)

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

    def shuffle(self, sf_ratio, sf_prob, reshuffle=True):
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


if __name__ == "__main__":
    for n_layer in [20, 32, 44, 56, 110]:
        for pe_alpha in [0.0, 0.01, 0.05, 0.1]:
            model = CifarPEResNetGN(
                pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
                pe_ratio=1.0, pe_t=1.0, n_layer=n_layer, n_classes=10
            )
            model.eval()

            xs = torch.randn(20, 3, 32, 32)
            out1 = model(xs)[1]
            out2 = model(xs)[1]

            model.shuffle(sf_ratio=1.0, sf_prob=1.0)
            out3 = model(xs)[1]

            print(torch.abs(out1 - out2).mean())
            print(torch.abs(out1 - out3).mean())

            xs = torch.randn(20, 3, 32, 32)
            out1 = model(xs)[1]
            out2 = model(xs)[1]

            model.shuffle(sf_ratio=1.0, sf_prob=1.0)
            out3 = model(xs)[1]

            print(torch.abs(out1 - out2).mean())
            print(torch.abs(out1 - out3).mean())

    for n_layer in [18, 34, 50, 101]:
        for pe_alpha in [0.0, 0.01, 0.05, 0.1]:
            model = SelfPEResNetGN(
                pe_way="sin", pe_op="add", pe_alpha=pe_alpha,
                pe_ratio=1.0, pe_t=1.0, n_layer=n_layer, n_classes=10
            )
            model.eval()

            xs = torch.randn(20, 3, 32, 32)
            out1 = model(xs)[1]
            out2 = model(xs)[1]

            model.shuffle(sf_ratio=1.0, sf_prob=1.0)
            out3 = model(xs)[1]

            print(torch.abs(out1 - out2).mean())
            print(torch.abs(out1 - out3).mean())

            xs = torch.randn(20, 3, 32, 32)
            out1 = model(xs)[1]
            out2 = model(xs)[1]

            model.shuffle(sf_ratio=1.0, sf_prob=1.0)
            out3 = model(xs)[1]

            print(torch.abs(out1 - out2).mean())
            print(torch.abs(out1 - out3).mean())

