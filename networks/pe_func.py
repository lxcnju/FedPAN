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
