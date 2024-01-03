# Le Thanh Hai
# June 20, 2023
# modify the pretrained models from 3 channels to 36 channels
import torch
import torch.nn as nn
import torchvision.models as models
from types import SimpleNamespace


class pretrained_models(nn.Module):
    # Load the pretrained models and modify input channels of the first layer from 3 channels to 36 channels
    # and number of classes of the last layer to 1 for binary classification
    # def __init__(self, model_name, in_channels, out_channels, num_classes, kernel_size, stride, bias):
    def __init__(self, model_name, in_channels, num_classes):
        super(pretrained_models, self).__init__()
        self.hparams = SimpleNamespace(model_name=model_name,
                                       in_channels=in_channels,
                                       # out_channels=out_channels,
                                       num_classes=num_classes,
                                       # kernel_size=kernel_size,
                                       # stride=stride,
                                       # bias=bias
                                       )

        if self.hparams.model_name == 'densenet121':
            self.model = self.densenet121(models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
                                          self.hparams)
        if self.hparams.model_name == 'efficientnet':
            self.model = self.efficientnet(models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1),
                                           self.hparams)

    def get_1st_layer(self, first_layer):
        """
        pretrained_weights = model.features[0].weight
        out_channels = model.features[0].out_channels
        kernel_size = model.features[0].kernel_size
        stride = model.features[0].stride
        padding = model.features[0].padding
        print(out_channels, kernel_size, stride, padding)
        Or
        pretrained_weights = model.features.conv0.weight
        out_channels = model.features.conv0.out_channels
        kernel_size = model.features.conv0.kernel_size
        stride = model.features.conv0.stride
        padding = model.features.conv0.padding
        bias = model.features.conv0.bias
        print(kernel_size, stride, padding)
        """
        # get parameters of first layer (Conv2d layer) of model
        weights = first_layer.weight
        out_channels = first_layer.out_channels
        kernel_size = first_layer.kernel_size
        stride = first_layer.stride
        padding = first_layer.padding

        return weights, out_channels, kernel_size, stride, padding

    def get_last_layer(self, last_layer):
        # get in_features parameter of last layer (Linear layer) of model
        in_features = last_layer.in_features

        return in_features

    def densenet121(self, model, hparams):
        # get the pre-trained weights of the first layer
        # print(model)
        weights, out_channels, kernel_size, stride, padding = self.get_1st_layer(model.features[0])
        in_features = self.get_last_layer(model.classifier)
        new_features = nn.Sequential(*list(model.features.children()))
        # out_channels: 64 for densenet121, 96 for densenet161
        new_features[0] = nn.Conv2d(in_channels=hparams.in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=False
                                    )
        # For M-channel, weight should randomly initialized with Gaussian
        # new_features[0].weight.data.normal_(0, 0.001)
        nn.init.kaiming_uniform_(new_features[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        # For RGB it should be copied from pretrained weights
        new_features[0].weight.data[:, :3, :, :] = nn.Parameter(weights)
        model.features = new_features
        model.classifier = nn.Linear(in_features=in_features, out_features=hparams.num_classes)

        return model

    def efficientnet(self, model, hparams):
        # print(model)
        weights, out_channels, kernel_size, stride, padding = self.get_1st_layer(model.features[0][0])
        in_features = self.get_last_layer(model.classifier[1])
        new_features = nn.Sequential(*list(model.features.children()))
        # out_channels: 64 for densenet121, 96 for densenet161
        new_features[0][0] = nn.Conv2d(in_channels=hparams.in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       bias=False
                                       )
        # For M-channel, weight should randomly initialized with Gaussian
        # new_features[0].weight.data.normal_(0, 0.001)
        nn.init.kaiming_uniform_(new_features[0][0].weight, mode='fan_in', nonlinearity='leaky_relu')
        # For RGB it should be copied from pretrained weights
        new_features[0][0].weight.data[:, :3, :, :] = nn.Parameter(weights)
        model.features = new_features
        model.classifier[1] = nn.Linear(in_features=in_features, out_features=hparams.num_classes)

        return model

    def forward(self, x):
        out = self.model(x)

        return out


# def load_densenet121(in_channels=36, out_channels=64, num_classes=10, kernel_size=7, stride=2, bias=True):
def load_densenet121(in_channels=3, num_classes=10):
    # return pretrained_models(model_name='densenet121', in_channels=in_channels, out_channels=out_channels,
    #                          num_classes=num_classes, kernel_size=kernel_size, stride=stride, bias=bias)
    return pretrained_models(model_name='densenet121', in_channels=in_channels, num_classes=num_classes)


def load_efficientnet(in_channels=3, num_classes=10):
    return pretrained_models(model_name='efficientnet', in_channels=in_channels, num_classes=num_classes)

