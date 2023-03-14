from networks.mlp import MLPPENet
from networks.vgg import VGGPENet
from networks.resnet import CifarPEResNet
from networks.resnet_gn import CifarPEResNetGN

from utils import weights_init


def load_model(args):
    if args.net == "mlp":
        model = MLPPENet(
            input_size=args.input_size,
            n_hidden=args.n_hidden,
            n_classes=args.n_classes,
            pe_way=args.pe_way,
            pe_t=args.pe_t,
            pe_ratio=args.pe_ratio,
            pe_alpha=args.pe_alpha,
            pe_op=args.pe_op
        )
    elif args.net == "vgg8":
        model = VGGPENet(
            n_layer=8,
            n_classes=args.n_classes,
            pe_way=args.pe_way,
            pe_t=args.pe_t,
            pe_ratio=args.pe_ratio,
            pe_alpha=args.pe_alpha,
            pe_op=args.pe_op
        )
    elif args.net == "vgg11":
        model = VGGPENet(
            n_layer=11,
            n_classes=args.n_classes,
            pe_way=args.pe_way,
            pe_t=args.pe_t,
            pe_ratio=args.pe_ratio,
            pe_alpha=args.pe_alpha,
            pe_op=args.pe_op
        )
    elif args.net == "vgg13":
        model = VGGPENet(
            n_layer=13,
            n_classes=args.n_classes,
            pe_way=args.pe_way,
            pe_t=args.pe_t,
            pe_ratio=args.pe_ratio,
            pe_alpha=args.pe_alpha,
            pe_op=args.pe_op
        )
    elif args.net == "cifar_resnet20":
        model = CifarPEResNet(
            n_layer=20,
            n_classes=args.n_classes,
            pe_way=args.pe_way,
            pe_t=args.pe_t,
            pe_ratio=args.pe_ratio,
            pe_alpha=args.pe_alpha,
            pe_op=args.pe_op,
        )
    elif args.net == "cifar_resnet20_gn":
        model = CifarPEResNetGN(
            n_layer=20,
            n_classes=args.n_classes,
            pe_way=args.pe_way,
            pe_t=args.pe_t,
            pe_ratio=args.pe_ratio,
            pe_alpha=args.pe_alpha,
            pe_op=args.pe_op,
        )
    else:
        raise ValueError("No such net: {}".format(args.net))

    # initialization
    model.apply(weights_init)
    return model
