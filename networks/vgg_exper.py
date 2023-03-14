import torch
import torch.nn as nn
import torch.nn.functional as F

import random

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


class Reshape(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, xs):
        return xs.reshape((xs.shape[0], -1))


def get_vgg_cfg(n_layer):
    if n_layer == 8:
        cfg = [
            64, 'M',
            128, 'M',
            256, 'M',
            512, 'M',
            512, 'M'
        ]
    elif n_layer == 11:
        cfg = [
            64, 'M',
            128, 'M',
            256, 256, 'M',
            512, 512, 'M',
            512, 512, 'M'
        ]
    elif n_layer == 13:
        cfg = [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 'M',
            512, 512, 'M',
            512, 512, 'M'
        ]
    elif n_layer == 16:
        cfg = [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M'
        ]
    elif n_layer == 19:
        cfg = [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 256, 'M',
            512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'
        ]
    return cfg


def conv3x3(in_channel, out_channel):
    layer = nn.Conv2d(
        in_channel, out_channel,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    return layer


def make_layers(cfg, init_c=3):
    block = nn.ModuleList()

    in_c = init_c
    for e in cfg:
        if e == "M":
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            block.append(conv3x3(in_c, e))
            in_c = e

    return block


class VGGPENet(nn.Module):
    def __init__(
        self,
        n_layer=11,
        n_classes=10,
        pe_way="sin",
        pe_t=1.0,
        pe_alpha=1.0,
        pe_op="add",
        pe_ratio=1.0,
        pe_pos="bef_act",
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio
        self.pe_pos = pe_pos

        self.cfg = get_vgg_cfg(n_layer)
        self.layers = make_layers(self.cfg)
        self.layers.append(Reshape())
        self.layers.append(nn.Linear(512, n_classes))

        # layer ids need to add pe
        self.pe_ids = [
            i for i, layer in enumerate(self.layers) if isinstance(
                layer, nn.Conv2d
            )
        ]
        cfg_size = [
            e for e in self.cfg if isinstance(e, int)
        ]
        self.pe_sizes = cfg_size

        self.pes = self.generate_pes()

        # param layers that need to shuffle
        self.sf_sizes = cfg_size

        assert len(self.pe_ids) == len(self.sf_sizes)

        self.sf_infos = []
        for i, pe_id in enumerate(self.pe_ids):
            self.sf_infos.append((pe_id, "conv", (i, i - 1)))
        self.sf_infos.append((len(self.layers) - 1, "fc", (-1, i)))

    def generate_pes(self):
        pes = []
        for s, size in enumerate(self.pe_sizes):
            pe = sin_pe_func(
                pe_op=self.pe_op,
                pe_t=self.pe_t,
                pe_ratio=self.pe_ratio,
                pe_alpha=self.pe_alpha,
                n_hidden=size
            )
            pe = pe.reshape((1, -1, 1, 1))
            pes.append(pe)
        return pes

    def generate_sf_inds(self, sf_ratio, sf_prob):
        sf_inds = []
        for s, size in enumerate(self.sf_sizes):
            # inds = torch.randperm(size)
            nsf = int(size * sf_ratio)

            inds = list(range(size))
            for i in range(size):
                j = random.choice(range(i, min(size, i + nsf + 1)))

                if random.random() <= sf_prob:
                    temp = inds[i]
                    inds[i] = inds[j]
                    inds[j] = temp

            inds = torch.LongTensor(inds)
            sf_inds.append(inds)
        return sf_inds

    def forward(self, xs):
        hs = xs

        all_hs = [hs]
        for i, layer in enumerate(self.layers):
            if i + 1 == len(self.layers):
                hs0 = hs

            hs = layer(hs)

            if isinstance(layer, nn.Conv2d):
                if self.pe_pos == "aft_act":
                    hs = F.relu(hs)

            if i in self.pe_ids:
                mask = self.pes[self.pe_ids.index(i)]
                mask = mask.to(device=xs.device)

                if self.pe_op == "add":
                    hs = hs + mask
                elif self.pe_op == "mul":
                    hs = hs * mask
                else:
                    pass

            if isinstance(layer, nn.Conv2d):
                if self.pe_pos == "bef_act":
                    hs = F.relu(hs)

            all_hs.append(hs)

        return all_hs, hs0, hs

    def shuffle(self, sf_ratio, sf_prob, reshuffle=True):
        if reshuffle is True:
            self.sf_inds = self.generate_sf_inds(
                sf_ratio=sf_ratio, sf_prob=sf_prob
            )

        for la_id, la_type, la_inds in self.sf_infos:
            # dim0 shuffle
            if la_inds[0] != -1:
                inds = self.sf_inds[la_inds[0]]
                ws = self.layers[la_id].weight.data
                self.layers[la_id].weight.data = ws[inds]

                bs = self.layers[la_id].bias.data
                self.layers[la_id].bias.data = bs[inds]

            # dim1 shuffle
            if la_inds[1] != -1:
                inds = self.sf_inds[la_inds[1]]
                ws = self.layers[la_id].weight.data
                self.layers[la_id].weight.data = ws[:, inds]


if __name__ == "__main__":
    all_infos = []
    for pe_alpha in [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        net = VGGPENet(
            n_layer=13, n_classes=10,
            pe_way="sin", pe_t=1.0, pe_ratio=1.0,
            pe_alpha=pe_alpha, pe_op="add"
        )

        xs = torch.randn(32, 3, 32, 32)
        all_hs, hs0, hs = net(xs)

        infos = []
        for hs in all_hs:
            abs_value = torch.mean(hs).detach().numpy()
            std_value = torch.std(hs).detach().numpy()

            infos.append([abs_value, std_value])

        all_infos.append(infos)

    import matplotlib.pyplot as plt

    res = np.array(all_infos)
    print(res.shape)

    res = res.transpose(2, 1, 0)

    plt.figure(figsize=(20, 8))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            plt.subplot(res.shape[0], res.shape[1], i * res.shape[1] + j + 1)
            ys = res[i][j]

            plt.bar(range(len(ys)), ys)

    plt.show()

    res = np.array(all_infos)
    print(res.shape)

    res = res.transpose(2, 0, 1)

    plt.figure(figsize=(20, 8))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            plt.subplot(res.shape[0], res.shape[1], i * res.shape[1] + j + 1)
            ys = res[i][j]

            plt.bar(range(len(ys)), ys)

    plt.show()
