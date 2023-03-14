import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from paths import pretrain_fpaths


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, widen_factor=1, dropout_rate=0.0, stride=1
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=1,
            padding=1, bias=True
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, widen_factor=1, dropout_rate=0.0, stride=1
    ):
        super(Bottleneck, self).__init__()
        width = planes * widen_factor

        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3,
                               stride=stride, padding=1, bias=False)
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


class CifarWideResNet(nn.Module):
    def __init__(
        self, n_layer=28, widen_factor=2, dropout_rate=0.0, n_classes=10
    ):
        super().__init__()
        self.n_layer = n_layer
        self.widen_factor = widen_factor
        self.n_classes = n_classes

        assert ((n_layer - 4) % 6 == 0), "WideResNet depth is 6n+4"
        n = int((n_layer - 4) / 6)
        k = widen_factor

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )

        self.cfg = (BasicBlock, (n, n, n))
        self.in_planes = 16

        self.layer1 = self._make_layer(
            block=self.cfg[0], planes=16 * k,
            dropout_rate=dropout_rate,
            stride=1, num_blocks=self.cfg[1][0],
        )
        self.layer2 = self._make_layer(
            block=self.cfg[0], planes=32 * k,
            dropout_rate=dropout_rate,
            stride=2, num_blocks=self.cfg[1][1],
        )
        self.layer3 = self._make_layer(
            block=self.cfg[0], planes=64 * k,
            dropout_rate=dropout_rate,
            stride=2, num_blocks=self.cfg[1][2],
        )

        self.bn1 = nn.BatchNorm2d(64 * k, momentum=0.9)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            64 * k * self.cfg[0].expansion, n_classes
        )

    def _make_layer(self, block, planes, dropout_rate, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    dropout_rate=dropout_rate,
                    stride=stride,
                )
            )
            self.in_planes = block.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))

        # print("CifarWideResNet", self.n_layer, x.shape, out.shape)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SelfWideResNet(nn.Module):
    def __init__(
        self, n_layer=18, widen_factor=2, dropout_rate=0.0, n_classes=10,
        first_conv="7x7-pool"
    ):
        super().__init__()
        self.n_layer = n_layer
        self.widen_factor = widen_factor
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

        if n_layer == 18:
            self.cfg = (BasicBlock, (2, 2, 2, 2))
        elif n_layer == 34:
            self.cfg = (BasicBlock, (3, 4, 6, 3))
        elif n_layer == 50:
            self.cfg = (Bottleneck, (3, 4, 6, 3))
        elif n_layer == 101:
            self.cfg = (Bottleneck, (3, 4, 23, 3))
        elif n_layer == 152:
            self.cfg = (Bottleneck, (3, 8, 36, 3))
        else:
            raise ValueError("No such layer: {}".format(n_layer))

        self.in_planes = 64

        self.layer1 = self._make_layer(
            block=self.cfg[0], planes=64,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate,
            stride=1, num_blocks=self.cfg[1][0],
        )
        self.layer2 = self._make_layer(
            block=self.cfg[0], planes=128,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate,
            stride=2, num_blocks=self.cfg[1][1],
        )
        self.layer3 = self._make_layer(
            block=self.cfg[0], planes=256,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate,
            stride=2, num_blocks=self.cfg[1][2],
        )
        self.layer4 = self._make_layer(
            block=self.cfg[0], planes=512,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate,
            stride=2, num_blocks=self.cfg[1][3],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            512 * self.cfg[0].expansion,
            n_classes
        )

    def _make_layer(
        self, block, planes, widen_factor, dropout_rate, stride, num_blocks
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    widen_factor=widen_factor,
                    dropout_rate=dropout_rate,
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

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # print("WideResNet", self.n_layer, x.shape, out.shape)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def load_torch_wide_resnet(n_layer=50, n_classes=10, pretrain=False):
    if n_layer == 50:
        model = models.wide_resnet50_2(False)
        n_hidden = 2048
    elif n_layer == 101:
        model = models.wide_resnet101_2(False)
        n_hidden = 2048
    else:
        raise ValueError("No such n_layer: {}".format(n_layer))

    if pretrain is True:
        name = "wide_resnet{}_2".format(n_layer)
        fpath = pretrain_fpaths[name]
        pretrained = torch.load(fpath)
        print(pretrained.keys())
        model.load_state_dict(pretrained, strict=True)
        print("Pretrained net loaded from: {}".format(fpath))

    model.fc = nn.Linear(n_hidden, n_classes)
    return model


if __name__ == "__main__":
    for n_layer in [28]:
        for factor in [1, 2, 4, 8, 10, 12]:
            model = CifarWideResNet(
                n_layer=n_layer, widen_factor=factor,
                dropout_rate=0.0, n_classes=10,
            )
            n_params = sum([
                param.numel() for param in model.parameters()
            ])
            print("WRN-{}-{}".format(n_layer, factor))
            print("Total number of parameters : {}".format(n_params))

            xs = torch.randn(1, 3, 32, 32)
            hs = model(xs)
            print(hs.shape)

    """
    for n_layer in [18, 34, 50, 101, 152]:
        for factor in [1, 2]:
            model = SelfWideResNet(
                n_layer=n_layer, widen_factor=factor,
                dropout_rate=0.0, n_classes=10,
            )
            n_params = sum([
                param.numel() for param in model.parameters()
            ])
            print("WRN-{}-{}".format(n_layer, factor))
            print("Total number of parameters : {}".format(n_params))

            xs = torch.randn(1, 3, 224, 224)
            hs = model(xs)
            print(hs.shape)
    """
