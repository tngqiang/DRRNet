import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 Bottleneck 层
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

# 定义 Transition 层
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

# 定义 DenseNet 网络
class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, growth_rate=32, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # 初始卷积层
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        
        # Dense Block 和 Transition Layer
        self.dense1 = self._make_dense_block(block, num_channels, num_blocks[0])
        num_channels += num_blocks[0] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense2 = self._make_dense_block(block, num_channels, num_blocks[1])
        num_channels += num_blocks[1] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans2 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense3 = self._make_dense_block(block, num_channels, num_blocks[2])
        num_channels += num_blocks[2] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans3 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense4 = self._make_dense_block(block, num_channels, num_blocks[3])
        num_channels += num_blocks[3] * growth_rate
        
        # 全局平均池化和分类器
        self.bn = nn.BatchNorm2d(num_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, block, in_channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block(in_channels + i * self.growth_rate, self.growth_rate, bn_size=4))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        
        out = F.relu(self.bn(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 创建不同版本的 DenseNet
def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16])

def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32])

def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32])

def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)

# 测试网络
def test():
    net = DenseNet121()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())

# 运行测试
if __name__ == "__main__":
    test()    