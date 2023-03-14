import torch
import torch.nn as nn
import numpy as np
import random

from networks.pe_func import sin_pe_func


class MLPPENet(nn.Module):
    def __init__(
            self, input_size=784, n_hidden=1024, n_classes=10,
            pe_way="sin", pe_t=1.0, pe_ratio=1.0, pe_alpha=1.0, pe_op="add"):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.pe_way = pe_way
        self.pe_t = pe_t
        self.pe_alpha = pe_alpha
        self.pe_op = pe_op
        self.pe_ratio = pe_ratio

        layers = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_classes)
        )

        # add each layer
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)

        # layer ids need to add pe
        self.pe_ids = [0, 2, 4]
        self.pe_sizes = [
            n_hidden, n_hidden, n_hidden
        ]

        self.pes = self.generate_pes()

        # param layers that need to shuffle
        self.sf_sizes = [n_hidden, n_hidden, n_hidden]
        self.sf_infos = [
            (0, "fc", (0, -1)),
            (2, "fc", (1, 0)),
            (4, "fc", (2, 1)),
            (6, "fc", (-1, 2))
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
                mat = mats[la_inds[0]]
                ws = self.layers[la_id].weight.data
                mat = torch.FloatTensor(mat).to(ws.device)
                self.layers[la_id].weight.data = mat.mm(ws)

                bs = self.layers[la_id].bias.data.unsqueeze(dim=1)
                self.layers[la_id].bias.data = mat.mm(bs).squeeze(dim=1)

            # dim1 shuffle
            if la_inds[1] != -1:
                mat = mats[la_inds[1]]
                ws = self.layers[la_id].weight.data
                mat = torch.FloatTensor(mat).to(ws.device)
                self.layers[la_id].weight.data = ws.mm(mat.transpose(0, 1))

    def get_neuron_reps(self, xs, ys, rep_way, layer_id):
        assert layer_id in [0, 1, 2, 3]

        if rep_way == "weight":
            rep = self.layers[2 * layer_id].weight.data.cpu().numpy()
        elif rep_way == "activation":
            hs = xs.view((xs.shape[0], -1))

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

                if i == 2 * layer_id:
                    rep = hs.detach().cpu().numpy().transpose()
                    break

        elif rep_way == "preference":
            p_vec = []
            for c in range(self.n_classes):
                hs = xs.view((xs.shape[0], -1))

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

                    if i == 2 * layer_id:
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
                ).detach().cpu().numpy()
                p_vec.append(pc_vec)
            rep = np.stack(p_vec, axis=0).transpose()

        return rep


if __name__ == "__main__":
    for pe_alpha in [0.0, 0.01, 0.05, 0.1]:
        net = MLPPENet(
            input_size=784, n_hidden=1024, n_classes=10,
            pe_way="sin", pe_t=1.0, pe_ratio=1.0,
            pe_alpha=pe_alpha, pe_op="add"
        )

        xs = torch.randn(32, 784)
        out1 = net(xs)

        net.shuffle()
        out2 = net(xs)

        print(torch.abs(out1 - out2).mean())

