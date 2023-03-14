import os
import random
from collections import namedtuple
import numpy as np

import torch

from feddata import FedData
from algorithms.fedavg import FedAvg
from networks.model import load_model

from paths import save_dir
from config import default_param_dicts

torch.set_default_tensor_type(torch.FloatTensor)


def get_one_hyper(algo):
    if algo == "fedavg":
        hypers = {
            "cnt": 1,
            "none": ["none"]
        }
    else:
        raise ValueError("No such fed algo:{}".format(algo))
    return hypers


def main_federated(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # DataSets
    try:
        n_clients = args.n_clients
    except Exception:
        n_clients = None

    try:
        nc_per_client = args.nc_per_client
    except Exception:
        nc_per_client = None

    try:
        dir_alpha = args.dir_alpha
    except Exception:
        dir_alpha = None

    feddata = FedData(
        dataset=args.dataset,
        split=args.split,
        n_clients=n_clients,
        nc_per_client=nc_per_client,
        dir_alpha=dir_alpha,
        n_max_sam=args.n_max_sam,
    )
    csets, gset = feddata.construct()

    try:
        nc = int(args.dset_ratio * len(csets))
        clients = list(csets.keys())
        sam_clients = np.random.choice(
            clients, nc, replace=False
        )
        csets = {
            c: info for c, info in csets.items() if c in sam_clients
        }
    except Exception:
        pass

    feddata.print_info(csets, gset)

    # Model
    model = load_model(args)
    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    algo = FedAvg(
        csets=csets,
        gset=gset,
        model=model,
        args=args
    )
    algo.train()

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main(algo, net, datasets):
    dir_alphas = [0.5]

    pe_pairs = [
        ("sin", 1.0, "add", 0.0),
        ("sin", 1.0, "mul", 0.05),
        ("sin", 4.0, "mul", 0.1),
        ("sin", 8.0, "mul", 0.15),
        ("sin", 1.0, "add", 0.15),
        ("sin", 4.0, "add", 0.25),
        ("sin", 8.0, "add", 0.5),
    ]

    optim_pairs = [
        ("SGD", 0.9, 0.05),
    ]

    hypers = get_one_hyper(algo)

    for dataset in datasets:
        if dataset == "cifar100":
            max_round = 500
            n_classes = 100
        elif dataset == "cifar10":
            max_round = 500
            n_classes = 10

        for dir_alpha in dir_alphas:
            for pe_way, pe_t, pe_op, pe_alpha in pe_pairs:
                for optimizer, momentum, lr in optim_pairs:
                    for j in range(hypers["cnt"]):
                        para_dict = {}
                        for k, vs in default_param_dicts[dataset].items():
                            para_dict[k] = random.choice(vs)

                        para_dict["algo"] = algo
                        para_dict["dataset"] = dataset
                        para_dict["split"] = "dirichlet"
                        para_dict["dir_alpha"] = dir_alpha

                        para_dict["optimizer"] = optimizer
                        para_dict["momentum"] = momentum

                        if net == "vgg8":
                            para_dict["net"] = "vgg8"
                            para_dict["n_layer"] = 8
                            para_dict["lr"] = 0.05
                        elif net == "vgg11":
                            para_dict["net"] = "vgg11"
                            para_dict["n_layer"] = 11
                            para_dict["lr"] = 0.03
                        elif net == "cifar_resnet20":
                            para_dict["net"] = "cifar_resnet20"
                            para_dict["n_layer"] = 20
                            para_dict["lr"] = 0.1
                        elif net == "cifar_resnet20_gn":
                            para_dict["net"] = "cifar_resnet20_gn"
                            para_dict["n_layer"] = 20
                            para_dict["lr"] = 0.03

                        para_dict["pe_way"] = pe_way
                        para_dict["pe_t"] = pe_t
                        para_dict["pe_op"] = pe_op
                        para_dict["pe_alpha"] = pe_alpha
                        para_dict["pe_ratio"] = 1.0

                        for key, values in hypers.items():
                            if key == "cnt":
                                continue
                            else:
                                para_dict[key] = values[j]

                        para_dict["max_round"] = max_round
                        para_dict["test_round"] = int(max_round / 20)
                        para_dict["n_classes"] = n_classes
                        para_dict["n_clients"] = 100
                        para_dict["c_ratio"] = 0.1
                        para_dict["local_epochs"] = 5

                        para_dict["fname"] = "{}-{}-pans.log".format(
                            algo, dataset
                        )

                        main_federated(para_dict)


if __name__ == "__main__":
    pairs = [
        ("fedavg", "cifar10", "vgg11"),
        ("fedavg", "cifar100", "cifar_resnet20"),
    ]
    for algo, dset, net in pairs:
        main(algo, net, [dset])
