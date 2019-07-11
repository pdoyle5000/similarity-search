import torch
import torch.nn as nn
import torch.nn.functional as F

def simple_layer(c_in, c_out, kernel):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, padding=(1, 1), stride=(1, 1)),
        nn.BatchNorm2d(c_out, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True))

def simple_layer_w_pool(c_in, c_out, kernel, pool_kernel):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, padding=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(
            kernel_size=pool_kernel,
            stride=pool_kernel,
            dilation=1,
            ceil_mode=False),
        nn.Dropout2d(p=0.1))


class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.conv1 = simple_layer(3, 64, 3)
        self.conv2 = simple_layer(64, 128, 3)
        self.conv3 = simple_layer(128, 128, 3)
        self.conv4 = simple_layer_w_pool(128, 128, 3, 2)
        self.conv5 = simple_layer(128, 128, 3)
        self.conv6 = simple_layer(128, 128, 3)
        self.conv7 = simple_layer_w_pool(128, 256, 3, 2)
        self.conv8 = simple_layer(256, 256, 3)
        self.conv9 = simple_layer_w_pool(256, 256, 3, 2)
        self.conv10 = simple_layer_w_pool(256, 512, 3, 2)
        self.conv11 = simple_layer(512, 2048, 1)
        self.conv12 = simple_layer_w_pool(2048, 256, 1, 2)
        self.conv13 = simple_layer(256, 256, 3)
        self.dense = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv13(out)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = F.dropout2d(out, 0.1, training=True)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return out


