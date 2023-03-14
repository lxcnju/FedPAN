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


class MLPPENet(nn.Module):
    def __init__(
            self, input_size=784, n_hidden=1024, n_classes=10,
            pe_way="sin", pe_t=1.0, pe_ratio=1.0, pe_alpha=1.0,
            pe_op="add", pe_pos="bef_act"):
        super().__init__()
        self.input_size = input_size
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio
        self.pe_pos = pe_pos

        layers = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_classes)
        )

        # add each layer
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)

        # layer ids need to add pe
        self.pe_ids = [0, 1, 2]
        self.pe_sizes = [
            n_hidden, n_hidden, n_hidden
        ]

        self.pes = self.generate_pes()

        # param layers that need to shuffle
        self.sf_sizes = [n_hidden, n_hidden, n_hidden]
        self.sf_infos = [
            (0, "fc", (0, -1)),
            (1, "fc", (1, 0)),
            (2, "fc", (2, 1)),
            (3, "fc", (-1, 2))
        ]

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
            pe = pe.reshape((1, -1))
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
        hs = xs.view((xs.shape[0], -1))

        all_hs = [hs]
        for i, layer in enumerate(self.layers):
            if i + 1 == len(self.layers):
                hs0 = hs

            hs = layer(hs)

            if self.pe_pos == "aft_act":
                if i + 1 != len(self.layers):
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

            if self.pe_pos == "bef_act":
                if i + 1 != len(self.layers):
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
        net = MLPPENet(
            input_size=784, n_hidden=1024, n_classes=10,
            pe_way="sin", pe_t=1.0, pe_ratio=1.0,
            pe_alpha=pe_alpha, pe_op="add"
        )

        xs = torch.randn(32, 784)
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
