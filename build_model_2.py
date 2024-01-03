import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, p=0.5):
        super(single_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),            
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class single_conv_noBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1):
        super(single_conv_noBN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding='same', bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class single_conv_mish(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, p=0.5):
        super(single_conv_mish, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding='same', bias=False),
            nn.Mish(),
            nn.BatchNorm2d(num_features=ch_out),
            nn.Dropout(p=p),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, p=0.5):
        super(double_conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding='same', bias=False),            
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding='same', bias=False),            
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class triple_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, p=0.5):
        super(triple_conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding='same', bias=False),            
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding='same', bias=False),            
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding='same', bias=False),            
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py
class SqEx(nn.Module):
    def __init__(self, n_features, reduction=16):  # 4, 8, 16
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


# Residual block using Squeeze and Excitation
# example of using SE block
class ResBlockSqEx(nn.Module):

    def __init__(self, n_features):
        super(ResBlockSqEx, self).__init__()
        # convolutions
        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(n_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        # squeeze and excitation
        self.sqex = SqEx(n_features)

    def forward(self, x):
        # convolutions
        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))

        # squeeze and excitation
        y = self.sqex(y)

        # add residuals
        y = torch.add(x, y)

        return y
    

class conv_mish_SEblock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, p=0.5):
        super(conv_mish_SEblock, self).__init__()
        self.conv = single_conv_noBN(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size)
        self.mish = nn.Sequential(
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=ch_out),
            nn.Dropout(p=p),
        )
        self.sqex = SqEx(n_features=ch_out)

    def forward(self, x):
        y1 = self.mish(self.conv(x))
        # squeeze and excitation
        y2 = self.sqex(y1)
        y = torch.add(y1, y2)

        return y


class conv_SEblock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, p=0.5):
        super(conv_SEblock, self).__init__()
        self.conv = single_conv_noBN(ch_in=ch_in, ch_out=ch_out, kernel_size=kernel_size)
        self.act = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=ch_out),
            nn.Dropout(p=p),
        )
        self.sqex = SqEx(n_features=ch_out)

    def forward(self, x):
        y1 = self.act(self.conv(x))
        # squeeze and excitation
        y2 = self.sqex(y1)
        y = torch.add(y1, y2)

        return y


# Define a convolution neural network
def _init_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(mean=0.0, std=0.15)  # small is better
        if module.bias is not None:
            module.bias.data.zero_()


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('Initialize network with %s' % init_type)
    net.apply(init_func)


class CNN(nn.Module):
    # like VGG
    def __init__(self, input_channels=3, num_classes=10, n_filters=128):
        super(CNN, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2)  # default: stride=kernel_size(=2)

        self.Conv1 = single_conv(ch_in=input_channels, ch_out=n_filters, p=0)
        self.Conv2 = double_conv_block(ch_in=n_filters, ch_out=n_filters * 2, p=0)
        self.Conv3 = triple_conv_block(ch_in=n_filters * 2, ch_out=n_filters * 4, p=0)
        self.Conv4 = single_conv(ch_in=n_filters * 4, ch_out=n_filters * 4, p=0)        

        self.Avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),            
            nn.Linear(in_features=n_filters * 4, out_features=n_filters * 2),
            nn.BatchNorm1d(num_features=n_filters * 2),
            # nn.Dropout(0.2),
            nn.Linear(in_features=n_filters * 2, out_features=num_classes),
        )
        
        self.apply(init_weights)

    def forward(self, input):
        output = self.Conv1(input)
        output = self.Maxpool(output)
        output = self.Conv2(output)
        output = self.Maxpool(output)
        output = self.Conv3(output)
        output = self.Maxpool(output)
        output = self.Conv4(output)        
        output = self.Avgpool(output)        
        output = self.Classifier(output)
        
        return output


class CNN_2(nn.Module):
    # Ref: STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
    # https://arxiv.org/pdf/1412.6806.pdf
    def __init__(self, input_channels=3, num_classes=10, n_filters=128):
        super(CNN_2, self).__init__()

        self.Conv1 = double_conv_block(ch_in=input_channels, ch_out=n_filters, p=0)
        # replace Max pooling layer by Conv with stride=2 to reduce dimensions, 112
        self.Conv2 = single_conv(ch_in=n_filters, ch_out=n_filters, stride=2, p=0)
        self.Conv3 = double_conv_block(ch_in=n_filters, ch_out=n_filters * 2, p=0)
        self.Conv4 = single_conv(ch_in=n_filters * 2, ch_out=n_filters * 2, stride=2, p=0)  # replace Max pooling layer, 56
        self.Conv5 = double_conv_block(ch_in=n_filters * 2, ch_out=n_filters * 4, p=0)
        self.Conv6 = single_conv(ch_in=n_filters * 4, ch_out=n_filters * 4, stride=2, p=0)  # replace Max pooling layer, 28
        self.Conv7 = double_conv_block(ch_in=n_filters * 4, ch_out=n_filters * 8, p=0)
        self.Conv8 = single_conv(ch_in=n_filters * 8, ch_out=n_filters * 8, stride=2, p=0)  # replace Max pooling layer, 14
        
        self.Conv11 = single_conv(ch_in=n_filters*8, ch_out=n_filters*8, p=0)
        self.Conv12 = single_conv(ch_in=n_filters*8, ch_out=n_filters*8, kernel_size=1, p=0)  # replace Flatten

        self.Conv13 = single_conv(ch_in=n_filters*8, ch_out=num_classes, kernel_size=1, p=0)        
        self.Classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            self.Conv13,            
        )
        
        self.apply(init_weights)

    def forward(self, input):
        output = self.Conv1(input)
        output = self.Conv2(output)
        output = self.Conv3(output)
        output = self.Conv4(output)
        output = self.Conv5(output)
        output = self.Conv6(output)
        output = self.Conv7(output)
        output = self.Conv8(output)        
        output = self.Conv11(output)
        output = self.Conv12(output)        
        output = self.Classifier(output)  # output.shape = B * C * H * W
        output = nn.functional.avg_pool2d(output, output.size()[2:])  # output.shape = B * C * 1 * 1        
        # output size is B * C * 1 * 1 with C of 1 because of binary classification
        # so if apply squeeze, output will be only B. Then output need to be applied a reshape function not a squeeze
        output = output.reshape((output.size()[0], output.size()[1]))  # with output.shape = B * 1
        
        return output


class CNN_2_2(nn.Module):
    # modify CNN_2 steps as same as CNN
    # Ref: STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
    # https://arxiv.org/pdf/1412.6806.pdf
    def __init__(self, input_channels=3, num_classes=10, n_filters=128):
        super(CNN_2_2, self).__init__()

        self.Conv1 = double_conv_block(ch_in=input_channels, ch_out=n_filters, p=0)
        # replace Max pooling layer by Conv with stride=2 to reduce dimensions, 112
        self.Conv2 = single_conv(ch_in=n_filters, ch_out=n_filters, stride=2, p=0)
        self.Conv3 = double_conv_block(ch_in=n_filters, ch_out=n_filters * 2, p=0)
        self.Conv4 = single_conv(ch_in=n_filters * 2, ch_out=n_filters * 2, stride=2,
                                 p=0)  # replace Max pooling layer, 56
        self.Conv5 = double_conv_block(ch_in=n_filters * 2, ch_out=n_filters * 4, p=0)
        self.Conv6 = single_conv(ch_in=n_filters * 4, ch_out=n_filters * 4, stride=2,
                                 p=0)  # replace Max pooling layer, 28
        self.Conv7 = single_conv(ch_in=n_filters * 4, ch_out=n_filters * 4, p=0)

        self.Conv8 = single_conv(ch_in=n_filters * 4, ch_out=n_filters * 4, kernel_size=1, p=0)  # replace Flatten

        self.Conv9 = single_conv(ch_in=n_filters * 4, ch_out=n_filters * 2, kernel_size=1, p=0)  # reduce to 256
        self.Conv10 = single_conv(ch_in=n_filters * 2, ch_out=num_classes, kernel_size=1, p=0)        
        self.Classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            self.Conv10,
        )

        # self.apply(_init_weights)
        self.apply(init_weights)

    def forward(self, input):
        output = self.Conv1(input)
        output = self.Conv2(output)
        output = self.Conv3(output)
        output = self.Conv4(output)
        output = self.Conv5(output)
        output = self.Conv6(output)
        output = self.Conv7(output)
        output = self.Conv8(output)        
        output = self.Conv9(output)        
        output = self.Classifier(output)  # output.shape = B * C * H * W        
        output = nn.functional.avg_pool2d(output, output.size()[2:])  # output.shape = B * C * 1 * 1                
        # output size is B * C * 1 * 1 with C of 1 because of binary classification
        # so if apply squeeze, output will be only B. Then output need to be applied a reshape function not a squeeze
        output = output.reshape((output.size()[0], output.size()[1]))  # with output.shape = B * 1
        
        return output


