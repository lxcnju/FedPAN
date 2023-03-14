import numpy as np


def combine_femnist(users_data):
    train_xs, train_ys = [], []
    test_xs, test_ys = [], []

    for client, info in users_data.items():
        n_test = max(int(0.2 * len(info["xs"])), 1)
        inds = np.random.permutation(info["xs"].shape[0])
        client_xs = info["xs"][inds]
        client_ys = info["ys"][inds]

        train_xs.append(client_xs[n_test:])
        train_ys.append(client_ys[n_test:])
        test_xs.append(client_xs[:n_test])
        test_ys.append(client_ys[:n_test])

    train_xs = np.concatenate(train_xs, axis=0)
    train_ys = np.concatenate(train_ys, axis=0)
    test_xs = np.concatenate(test_xs, axis=0)
    test_ys = np.concatenate(test_ys, axis=0)
    return train_xs, train_ys, test_xs, test_ys


def load_data(dset, args):
    if dset in ["cifar10", "cifar100"]:
        from datasets.cifar_data import load_cifar_data as load_data
        from datasets.cifar_data import CifarDataset as Dataset
    else:
        raise ValueError("No such dset: {}".format(dset))

    if dset == "femnist":
        users_data = load_data()

        nc = int(0.1 * len(users_data))
        clients = list(users_data.keys())
        sam_clients = np.random.choice(
            clients, nc, replace=False
        )
        users_data = {
            c: info for c, info in users_data.items() if c in sam_clients
        }

        train_xs, train_ys, test_xs, test_ys = combine_femnist(users_data)
    else:
        train_xs, train_ys, test_xs, test_ys = load_data(
            dset, combine=False
        )

    print(train_xs.shape, train_ys.shape, test_xs.shape, test_ys.shape)
    print(train_xs.max(), train_xs.min())

    train_set = Dataset(train_xs, train_ys, is_train=True, args=args)
    test_set = Dataset(test_xs, test_ys, is_train=False, args=args)
    return train_set, test_set
