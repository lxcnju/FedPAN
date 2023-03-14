
default_param_dicts = {
    "mnist": {
        "dataset": ["mnist"],
        "input_channel": [1],
        "input_size": [28],
        "split": ["label"],
        "n_classes": [10],
        "n_clients": [100],
        "nc_per_client": [2],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["MLPNet"],
        "max_round": [200],
        "test_round": [2],
        "local_epochs": [3],
        "local_steps": [None],
        "batch_size": [32],
        "optimizer": ["SGD"],
        "lr": [0.05],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "mnistm": {
        "dataset": ["mnistm"],
        "input_channel": [3],
        "input_size": [32],
        "split": ["label"],
        "n_classes": [10],
        "n_clients": [100],
        "nc_per_client": [2],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["LeNet"],
        "max_round": [200],
        "test_round": [2],
        "local_epochs": [3],
        "local_steps": [None],
        "batch_size": [32],
        "optimizer": ["SGD"],
        "lr": [0.05],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "svhn": {
        "dataset": ["svhn"],
        "input_channel": [3],
        "input_size": [32],
        "split": ["label"],
        "n_classes": [10],
        "n_clients": [100],
        "nc_per_client": [2],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["TFCNN"],
        "max_round": [200],
        "test_round": [2],
        "local_epochs": [3],
        "local_steps": [None],
        "batch_size": [32],
        "optimizer": ["SGD"],
        "lr": [0.05],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "cifar10": {
        "dataset": ["cifar10"],
        "split": ["dirichlet"],
        "dir_alpha": [1.0],
        "n_classes": [10],
        "n_clients": [100],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["VGG8"],
        "max_round": [1500],
        "test_round": [10],
        "local_epochs": [2],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "cinic10": {
        "dataset": ["cifar10"],
        "split": ["dirichlet"],
        "dir_alpha": [1.0],
        "n_classes": [10],
        "n_clients": [100],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["VGG8"],
        "max_round": [1500],
        "test_round": [10],
        "local_epochs": [2],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "cifar100": {
        "dataset": ["cifar100"],
        "split": ["dirichlet"],
        "dir_alpha": [1.0],
        "n_classes": [100],
        "n_clients": [100],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["VGG8"],
        "max_round": [1500],
        "test_round": [10],
        "local_epochs": [2],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "gtsrb": {
        "dataset": ["gtsrb"],
        "split": ["dirichlet"],
        "dir_alpha": [1.0],
        "n_classes": [43],
        "n_clients": [100],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["VGG8"],
        "max_round": [1500],
        "test_round": [10],
        "local_epochs": [5],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "ohsumed": {
        "dataset": ["ohsumed"],
        "split": ["dirichlet"],
        "dir_alpha": [1.0],
        "n_classes": [23],
        "n_clients": [100],
        "n_max_sam": [None],
        "c_ratio": [0.1],
        "net": ["OhsumedGRU"],
        "max_round": [100],
        "test_round": [1],
        "local_epochs": [4],
        "local_steps": [None],
        "batch_size": [32],
        "optimizer": ["SGD"],
        "lr": [0.1],
        "momentum": [0.9],
        "weight_decay": [1e-6],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "femnist": {
        "dataset": ["femnist"],
        "dset_ratio": [0.1],
        "split": ["user"],
        "n_classes": [62],
        "n_max_sam": [None],
        "c_ratio": [0.05],
        "net": ["FeMnistNet"],
        "max_round": [500],
        "test_round": [5],
        "local_epochs": [5],
        "local_steps": [None],
        "batch_size": [10],
        "optimizer": ["SGD"],
        "lr": [4e-3],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "shakespeare": {
        "dataset": ["shakespeare"],
        "split": ["user"],
        "dset_ratio": [0.1],
        "n_vocab": [81],
        "n_classes": [81],
        "n_max_sam": [None],
        "c_ratio": [0.05],
        "net": ["CharLSTM"],
        "max_round": [500],
        "test_round": [5],
        "local_epochs": [None],
        "local_steps": [50],
        "batch_size": [10],
        "optimizer": ["SGD"],
        "lr": [1.47],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "speechcommands": {
        "dataset": ["speechcommands"],
        "split": ["user"],
        "n_classes": [12],
        "n_max_sam": [None],
        "c_ratio": [0.01],
        "net": ["AudioDSCNN"],
        "max_round": [300],
        "test_round": [3],
        "local_steps": [50],
        "local_epochs": [None],
        "batch_size": [32],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
}