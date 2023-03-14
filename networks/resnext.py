import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from paths import pretrain_fpaths


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, groups=32, base_width=4, stride=1):
        super(Bottleneck, self).__init__()
        width = int(planes * groups * base_width / 64.0)

        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3,
            stride=stride, padding=1,
            groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNeXt(nn.Module):
    def __init__(self, n_layer=29, groups=8, base_width=64, n_classes=10):
        super().__init__()
        self.n_layer = n_layer
        self.groups = groups
        self.base_width = base_width
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        assert ((n_layer - 2) % 9 == 0), "CifarResNeXt depth is 9n+2"
        n = int((n_layer - 2) / 9)

        self.cfg = (Bottleneck, (n, n, n))
        self.in_planes = 64

        self.layer1 = self._make_layer(
            block=self.cfg[0], planes=64, groups=self.groups,
            base_width=self.base_width, stride=1, num_blocks=self.cfg[1][0],
        )
        self.layer2 = self._make_layer(
            block=self.cfg[0], planes=128, groups=self.groups,
            base_width=self.base_width, stride=2, num_blocks=self.cfg[1][1],
        )
        self.layer3 = self._make_layer(
            block=self.cfg[0], planes=256, groups=self.groups,
            base_width=self.base_width, stride=2, num_blocks=self.cfg[1][2],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            256 * self.cfg[0].expansion, n_classes
        )

    def _make_layer(
            self, block, planes, groups, base_width, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    groups=groups,
                    base_width=base_width,
                    stride=stride
                )
            )
            self.in_planes = block.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # print("CifarResNet", self.n_layer, x.shape, out.shape)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SelfResNeXt(nn.Module):
    def __init__(
        self, n_layer=18, groups=32, base_width=4,
        n_classes=10, first_conv="7x7-pool"
    ):
        super().__init__()
        self.n_layer = n_layer
        self.groups = groups
        self.base_width = base_width
        self.n_classes = n_classes
        self.first_conv = first_conv

        if first_conv == "7x7-pool":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7,
                stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(
                3, stride=2, padding=1
            )
        elif first_conv == "7x7-nopool":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7,
                stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
        elif first_conv == "3x3-1":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3,
                stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
        elif first_conv == "3x3-2":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3,
                stride=2, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
        else:
            raise ValueError("No such first conv: {}".format(first_conv))

        if n_layer == 50:
            self.cfg = (Bottleneck, (3, 4, 6, 3))
        elif n_layer == 101:
            self.cfg = (Bottleneck, (3, 4, 23, 3))
        elif n_layer == 152:
            self.cfg = (Bottleneck, (3, 8, 36, 3))
        else:
            raise ValueError("No such n_layer: {}".format(n_layer))

        self.in_planes = 64

        self.layer1 = self._make_layer(
            block=self.cfg[0], planes=64, groups=self.groups,
            base_width=self.base_width, stride=1, num_blocks=self.cfg[1][0],
        )
        self.layer2 = self._make_layer(
            block=self.cfg[0], planes=128, groups=self.groups,
            base_width=self.base_width, stride=2, num_blocks=self.cfg[1][1],
        )
        self.layer3 = self._make_layer(
            block=self.cfg[0], planes=256, groups=self.groups,
            base_width=self.base_width, stride=2, num_blocks=self.cfg[1][2],
        )
        self.layer4 = self._make_layer(
            block=self.cfg[0], planes=512, groups=self.groups,
            base_width=self.base_width, stride=2, num_blocks=self.cfg[1][3],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            512 * self.cfg[0].expansion, n_classes
        )

    def _make_layer(
        self, block, planes, groups, base_width, stride, num_blocks
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    groups=groups,
                    base_width=base_width,
                    stride=stride
                )
            )
            self.in_planes = block.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.first_conv == "7x7-pool":
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.pool1(out)
        elif self.first_conv == "7x7-nopool":
            out = F.relu(self.bn1(self.conv1(x)))
        elif self.first_conv in ["3x3-1", "3x3-2"]:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            raise ValueError("No such first conv: {}".format(self.first_conv))
        # print(out.shape)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # print("ResNet", self.n_layer, x.shape, out.shape)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def load_torch_resnext_download(
        n_layer, groups=32, download=False):
    if n_layer == 50:
        model = models.resnext50_32x4d(download)
        n_hidden = 2048
    elif n_layer == 101:
        model = models.resnext101_32x8d(download)
        n_hidden = 2048
    else:
        raise ValueError("No such n_layer: {}".format(n_layer))
    return model, n_hidden


def load_torch_resnext(
        n_layer=50, groups=32, n_classes=10, pretrain=False):
    """ base_width = 4 for ResNeXt50
        base_width = 8 for ResNeXt101
    """

    model, n_hidden = load_torch_resnext_download(
        n_layer, groups, False
    )

    if pretrain is True:
        try:
            print("Try loading...")
            if n_layer == 50:
                name = "resnext50_32x4d"
            elif n_layer == 101:
                name = "resnext101_32x8d"
            else:
                raise ValueError("No such layer: {}".format(n_layer))

            fpath = pretrain_fpaths[name]
            pretrained = torch.load(fpath)
            print(pretrained.keys())
            model.load_state_dict(pretrained, strict=True)
            print("Pretrained net loaded from: {}".format(fpath))
        except Exception:
            print("Try downloading...")
            model = load_torch_resnext_download(n_layer, True)
            print("Pretrained net downloaded from the internet network.")

    model.fc = nn.Linear(n_hidden, n_classes)
    # model.fc.out_features = n_classes
    return model


if __name__ == "__main__":
    pairs = [
        (1, 64),
        (2, 64),
        (4, 64),
        (8, 64),
        (16, 64),
        (16, 32),
        (32, 16),
        (64, 4),
        (128, 2),
    ]

    for n_layer in [29]:
        for groups, base_width in pairs:
            model = CifarResNeXt(
                n_layer=n_layer, groups=groups,
                base_width=base_width, n_classes=10
            )
            n_params = sum([
                param.numel() for param in model.parameters()
            ])
            print("Total number of parameters : {}".format(n_params))

            xs = torch.randn(1, 3, 32, 32)
            hs = model(xs)
            print(hs.shape)

    """
    for n_layer in [50, 101]:
        model = SelfResNeXt(
            n_layer=n_layer, groups=32, base_width=4, n_classes=10
        )
        n_params = sum([
            param.numel() for param in model.parameters()
        ])
        print("Total number of parameters : {}".format(n_params))

        xs = torch.randn(1, 3, 224, 224)
        hs = model(xs)
        print(hs.shape)

    for n_layer in [50, 101]:
        model = SelfResNeXt(
            n_layer=n_layer, groups=32, base_width=8, n_classes=10
        )
        n_params = sum([
            param.numel() for param in model.parameters()
        ])
        print("Total number of parameters : {}".format(n_params))

        xs = torch.randn(1, 3, 224, 224)
        hs = model(xs)
        print(hs.shape)

    for n_layer in [50, 101]:
        model = load_torch_resnext(
            n_layer=n_layer, n_classes=10, pretrain=False
        )
        n_params = sum([
            param.numel() for param in model.parameters()
        ])
        print("Total number of parameters : {}".format(n_params))

        xs = torch.randn(1, 3, 224, 224)
        hs = model(xs)
        print(hs.shape)

    from resnet import load_torch_resnet
    for n_layer in [50, 101]:
        model = load_torch_resnet(
            n_layer=n_layer, n_classes=10, pretrain=False
        )
        n_params = sum([
            param.numel() for param in model.parameters()
        ])
        print("Total number of parameters : {}".format(n_params))

        xs = torch.randn(1, 3, 224, 224)
        hs = model(xs)
        print(hs.shape)
    """
