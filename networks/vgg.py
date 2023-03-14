import torch
import torch.nn as nn
import numpy as np
import random

from networks.pe_func import sin_pe_func


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
            block.append(nn.ReLU(inplace=True))
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
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio

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

        print("PEIDS:", self.pe_ids)

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
        for i, layer in enumerate(self.layers):
            if i + 1 == len(self.layers):
                hs0 = hs

            hs = layer(hs)

            if i in self.pe_ids:
                mask = self.pes[self.pe_ids.index(i)]
                mask = mask.to(device=xs.device)

                if self.pe_op == "add":
                    hs = hs + mask
                elif self.pe_op == "mul":
                    hs = hs * mask
                else:
                    pass

        return hs0, hs

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

    def align(self, mats):
        assert len(mats) == len(self.sf_sizes)

        for la_id, la_type, la_inds in self.sf_infos:
            # dim0 shuffle
            if la_inds[0] != -1:
                if la_type == "conv":
                    mat = mats[la_inds[0]]
                    ws = self.layers[la_id].weight.data
                    Co, Ci, W, H = ws.shape
                    ws = ws.reshape(Co, -1)

                    mat = torch.FloatTensor(mat).to(ws.device)
                    self.layers[la_id].weight.data = mat.mm(ws).reshape(
                        Co, Ci, W, H
                    )

                    bs = self.layers[la_id].bias.data.unsqueeze(dim=1)
                    self.layers[la_id].bias.data = mat.mm(bs).squeeze(dim=1)
                elif la_type == "fc":
                    mat = mats[la_inds[0]]
                    ws = self.layers[la_id].weight.data
                    mat = torch.FloatTensor(mat).to(ws.device)
                    self.layers[la_id].weight.data = mat.mm(ws)

            # dim1 shuffle
            if la_inds[1] != -1:
                if la_type == "conv":
                    mat = mats[la_inds[1]]
                    ws = self.layers[la_id].weight.data

                    Co, Ci, W, H = ws.shape
                    ws = ws.permute(0, 2, 3, 1).reshape(-1, Ci)

                    mat = torch.FloatTensor(mat).to(ws.device)
                    self.layers[la_id].weight.data = ws.mm(
                        mat.transpose(0, 1)
                    ).reshape(Co, W, H, Ci).permute(0, 3, 1, 2)
                else:
                    mat = mats[la_inds[1]]
                    ws = self.layers[la_id].weight.data
                    mat = torch.FloatTensor(mat).to(ws.device)
                    self.layers[la_id].weight.data = ws.mm(
                        mat.transpose(0, 1)
                    )

    def get_neuron_reps(self, xs, ys, rep_way, layer_id):
        assert layer_id in self.pe_ids

        if rep_way == "weight":
            rep = self.layers[layer_id].weight.data.cpu().numpy()
            rep = rep.reshape(rep.shape[0], -1)
        elif rep_way == "activation":
            hs = xs
            for i, layer in enumerate(self.layers):
                hs = layer(hs)

                if i in self.pe_ids:
                    mask = self.pes[self.pe_ids.index(i)]
                    mask = mask.to(device=xs.device)

                    if self.pe_op == "add":
                        hs = hs + mask
                    elif self.pe_op == "mul":
                        hs = hs * mask
                    else:
                        pass

                if i == layer_id:
                    rep = hs.detach().cpu().numpy()
                    rep = rep.max(axis=-1).max(axis=-1).transpose()
                    break
        elif rep_way == "preference":
            p_vec = []
            for c in range(self.n_classes):
                hs = xs

                c_hs = hs[ys == c]
                c_ys = ys[ys == c]

                for i, layer in enumerate(self.layers):

                    c_hs = layer(c_hs)

                    if i in self.pe_ids:
                        mask = self.pes[self.pe_ids.index(i)]
                        mask = mask.to(device=xs.device)

                        if self.pe_op == "add":
                            c_hs = c_hs + mask
                        elif self.pe_op == "mul":
                            c_hs = c_hs * mask
                        else:
                            pass

                    if i == layer_id:
                        c_acts = c_hs.detach().cpu().numpy()
                        c_hs_param = nn.Parameter(
                            torch.FloatTensor(c_acts), requires_grad=True
                        ).to(hs.device)
                        c_hs_param.retain_grad()
                        c_hs = c_hs_param

                criterion = nn.CrossEntropyLoss()
                loss = criterion(c_hs, c_ys)
                self.zero_grad()
                loss.backward()
                c_grads = c_hs_param.grad

                pc_vec = (-1.0 * c_hs_param.detach() * c_grads).mean(
                    dim=0
                ).mean(dim=-1).mean(dim=-1).detach().cpu().numpy()
                p_vec.append(pc_vec)
            rep = np.stack(p_vec, axis=0).transpose()

        return rep


if __name__ == "__main__":
    for pe_alpha in [0.0, 0.01, 0.05, 0.1]:
        net = VGGPENet(
            n_layer=13, n_classes=10,
            pe_way="sin", pe_t=1.0, pe_ratio=1.0,
            pe_alpha=pe_alpha, pe_op="add"
        )

        # print(net)
        # print(net.pe_ids)
        # print(net.pe_sizes)
        # print(net.sf_sizes)
        # print(net.sf_infos)

        xs = torch.randn(32, 3, 32, 32)
        out1 = net(xs)

        net.shuffle()
        out2 = net(xs)

        print(torch.abs(out1 - out2).mean())

