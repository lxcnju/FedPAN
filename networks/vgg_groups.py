import torch
import torch.nn as nn

import random

from networks.pe_func import sin_pe_func


class Reshape(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, xs):
        return xs.reshape((xs.shape[0], -1))


def get_vgg_cfg(n_layer, gc):
    if n_layer == 8:
        base_cfg = [
            64, 'M',
            128, 'M',
            256, 'M',
        ]
        inter_c = 256
        group_cfg = [
            gc, 'M',
            gc, 'M'
        ]
    elif n_layer == 11:
        base_cfg = [
            64, 'M',
            128, 'M',
            256, 256, 'M',
        ]
        inter_c = 256
        group_cfg = [
            gc, gc, 'M',
            gc, gc, 'M'
        ]
    return base_cfg, inter_c, group_cfg


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


class VGGGroupPENet(nn.Module):
    def __init__(
        self,
        n_layer=11,
        n_groups=10,
        n_classes=10,
        pe_way="sin",
        pe_t=1.0,
        pe_alpha=1.0,
        pe_op="add",
        pe_ratio=1.0,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio

        if n_layer == 8:
            self.gc = 128
        elif n_layer == 11:
            self.gc = 160

        self.n_gcs = int(self.n_classes / self.n_groups)
        assert (self.n_gcs * self.n_groups) == self.n_classes

        self.base_cfg, self.inter_c, self.group_cfg = get_vgg_cfg(
            n_layer, gc=self.gc
        )
        self.base_layers = make_layers(self.base_cfg)

        self.group_layers = nn.ModuleList()
        for _ in range(n_groups):
            layers = make_layers(self.group_cfg, init_c=self.inter_c)
            layers.append(Reshape())
            layers.append(nn.Linear(self.gc, self.n_classes))
            self.group_layers.append(layers)

        # layer ids need to add pe
        self.pe_ids = [
            i for i, layer in enumerate(self.base_layers) if isinstance(
                layer, nn.Conv2d
            )
        ]
        cfg_size = [
            e for e in self.base_cfg if isinstance(e, int)
        ]
        self.pe_sizes = cfg_size

        self.pes = self.generate_pes()

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

    def forward(self, xs):
        hs = xs
        for i, layer in enumerate(self.base_layers):
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

        outputs = []
        for groups in self.group_layers:
            ghs = hs

            for layer in groups:
                ghs = layer(ghs)

            outputs.append(ghs)

        logits = torch.stack(outputs, dim=0).mean(dim=0)
        return None, logits


if __name__ == "__main__":
    net = VGGGroupPENet(
        n_layer=11, n_classes=10, n_groups=10,
        pe_way="sin", pe_t=1.0, pe_ratio=1.0,
        pe_alpha=0.0, pe_op="add"
    )

    xs = torch.randn(32, 3, 32, 32)
    outputs = net(xs)

    for ghs in outputs:
        print(ghs.shape)
