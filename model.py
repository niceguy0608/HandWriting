import numpy as np

import torch.nn as nn
import torch.nn.functional as F  # 函数式编程的方法
from torchsummary import summary


def conv_dw(inp, oup, stride):  # 深度可分离卷积 ，当通道数较多的时候，使用该卷积方式可以大大简短卷积需要的时间，减少参数的数量，提高卷积效率
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 分组卷积 group=inp,则是对输入feature map进行分组，然后每组分别卷积  即实现了深度可分离卷积的第一步：逐通道卷积，通道之间信息不交互。
        nn.BatchNorm2d(inp),#对输入的四维数组进行批量标准化处理，参数为输入的通道数量
        nn.ReLU6(inplace=True),#对relu的输出值进行限制，不让其超过6
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),  # 第二部：逐点卷积
        nn.BatchNorm2d(oup),
    )


def conv_bn(inp, oup, stride):  # 普通卷积
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  # 卷积层 Conv2d 对由多个输入平面组成的输入信号进行二维卷积 3为卷积核尺寸 1为padding填充  调用函数就好，不需要自己去写卷积具体操作
        nn.BatchNorm2d(oup),  # 在训练过程中，数据分布会发生变化，对下一层网络的学习带来困难。Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。
        nn.ReLU6(inplace=True)  # 激励函数层 inplace为True，将计算得到的值直接覆盖之前的值
    )


class ConvNet(nn.Module):# 我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法。
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()  #首先找到Net的父类nn.Module，然后把类Net的对象self转换为类nn.Module的对象，然后“被转换”的类nn.Module对象调用自己的init函数进行初始化
        self.conv1 = conv_bn(3, 8, 1)  # 64x64x8 表示 输入图像 三个通道x[ , ,0],x[ , ,1],x[ , ,2]，且每一个通道都是64*64
        self.conv2 = conv_bn(8, 16, 1)  # 64x64x16
        self.conv3 = conv_dw(16, 32, 1)  # 64*64*32
        self.conv4 = conv_dw(32, 32, 2)  #32*32*32
        self.conv5 = conv_dw(32, 64, 1) #64*32*32
        self.conv6 = conv_dw(64, 64, 2)
        self.conv7 = conv_dw(64, 128, 1)
        self.conv8 = conv_dw(128, 128, 1)
        self.conv9 = conv_dw(128, 128, 1)
        self.conv10 = conv_dw(128, 128, 1)
        self.conv11 = conv_dw(128, 128, 1)
        self.conv12 = conv_dw(128, 256, 2)
        self.conv13 = conv_dw(256, 256, 1)
        self.conv14 = conv_dw(256, 256, 1)
        self.conv15 = conv_dw(256, 512, 2)
        self.conv16 = conv_dw(512, 512, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),  # 输入神经元个数 512*4*4 输出神经元个数 1024  这是神经网络的线性层
            nn.Dropout(0.2),  # 这个操作表示使每个位置的元素都有一定概率归0，以此来模拟现实生活中的某些频道的数据缺失，以达到数据增强的目的，以达到减少过拟合的效果
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self.weight_init()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x9 = F.relu(x8 + x9)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x11 = F.relu(x10 + x11)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x14 = F.relu(x13 + x14)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x = x16.view(x16.size(0), -1)  # 相当于reshape
        x = self.classifier(x)
        return x

    def weight_init(self):
        for layer in self.modules():
            self._layer_init(layer)

    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


if __name__ == "__main__":
    model = ConvNet(3755).cuda()# 3755 表示特征数为3755
    summary(model, input_size=(3, 64, 64), device='cuda')
