import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm_layer=None):
        super(wide_basic, self).__init__()
        if norm_layer is None:
            # self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=True, momentum=0.1)  # also check with momentum=1.0
        else:
            self.bn1 = norm_layer(num_channels=in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.dropout = nn.Dropout(p=dropout_rate)

        if norm_layer is None:
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True, momentum=0.1)
        else:
            self.bn2 = norm_layer(num_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        # out = self.conv1(x)
        #out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv2(out)
        if self.shortcut is not None:
            out += self.shortcut(x)
        return out


class Wide_ResNet(nn.Module):
    def __init__(self, num_layers, depth, widen_factor, dropout_rate, num_classes,
                 num_input_channels=3, norm_layer=None):
        super(Wide_ResNet, self).__init__()
        self.num_layers = num_layers

        assert ((depth-4)%6 ==0), 'wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| wide-resnet %dx%d' %(depth, k))
        nstages = [16*k, 32*k, 64*k, 128*k]
        self.in_planes = nstages[0]

        self.conv1 = conv3x3(num_input_channels, nstages[0])
        if self.num_layers >= 1:
            self.layer1 = self._wide_layer(wide_basic, nstages[1], n, dropout_rate, stride=1, norm_layer=norm_layer)
        if self.num_layers >= 2:
            self.layer2 = self._wide_layer(wide_basic, nstages[2], n, dropout_rate, stride=2, norm_layer=norm_layer)
        if self.num_layers == 3:
            self.layer3 = self._wide_layer(wide_basic, nstages[3], n, dropout_rate, stride=2, norm_layer=norm_layer)
        if norm_layer is None:
            self.bn1 = nn.BatchNorm2d(nstages[num_layers], track_running_stats=True, momentum=0.1)
        else:
            self.bn1 = norm_layer(num_channels=nstages[num_layers])
        self.linear = nn.Linear(nstages[num_layers], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, norm_layer=None):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, norm_layer))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, input_dict):
        x = input_dict['inputs']
        out = self.conv1(x)
        if self.num_layers == 1:
            out1 = self.layer1(out)
            out = F.relu(self.bn1(out1))
            # out = F.relu(out1)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            emb = out.view(out.size(0), -1)
            out = self.linear(emb)

            output = out, [out1]
        elif self.num_layers == 2:
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out = F.relu(self.bn1(out2))
            # out = F.relu(out2)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            emb = out.view(out.size(0), -1)
            out = self.linear(emb)

            output = out, [out1, out2]
        elif self.num_layers == 3:
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out = F.relu(self.bn1(out3))
            out = F.adaptive_avg_pool2d(out, (1, 1))
            emb = out.view(out.size(0), -1)
            out = self.linear(emb)

            output = out, [out1, out2, out3]
        else:
            print('Specify valid number of layers. Now it is {}'.format(self.num_layers)); exit()

        return output


def WideResNet1BL(num_classes = 10):
    return Wide_ResNet(1, 28, 10, 0., num_classes)


def WideResNet2BL(num_classes = 10):
    return Wide_ResNet(2, 34, 10, 0., num_classes)
