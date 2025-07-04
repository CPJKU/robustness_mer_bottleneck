from torch import nn as nn


class PreConceptModel(nn.Module):
    """ Model architecture up to linear (prediction) layer(s). """
    def __init__(self):
        super(PreConceptModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, 2), # (in_channels, out_channels, kernel_size, stride, padding)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.mp2x2_dropout = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.conv7b = nn.Sequential(
            nn.Conv2d(384, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        """ Forward pass. """
        # 313 * 149 * 1
        x = self.conv1(x)           # 157 * 75 * 64
        x = self.conv2(x)           # 157 * 75 * 64
        x = self.mp2x2_dropout(x)   # 78 * 37 * 64
        x = self.conv3(x)           # 78 * 37 * 128
        x = self.conv4(x)           # 78 * 37 * 128
        x = self.mp2x2_dropout(x)   # 39 * 18 * 128
        x = self.conv5(x)           # 39 * 18 * 256
        x = self.conv6(x)           # 39 * 18 * 256
        x = self.conv7(x)           # 39 * 18 * 384
        x = self.conv7b(x)          # 39 * 18 * 384
        x = self.conv11(x)          # 2 * 2 * 256
        x = self.pool(x)
        return x

class Audio2Target(nn.Module):
    """ Architecture of standard model (PreConceptModel and one linear layer). """
    def __init__(self, num_targets, initialize=True):
        super(Audio2Target, self).__init__()

        self.preconcept_model = PreConceptModel()
        self.fc = nn.Linear(256, num_targets)

        # initialise weights
        if initialize:
            self.apply(initialize_weights)

    def forward(self, x):
        """ Forward pass. """
        x = self.preconcept_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return  {"output": x, 'c_prob': None}


class Audio2Ml2Emo(nn.Module):
    """ Architecture of bottleneck model (PreConceptModel and two linear layers). """
    def __init__(self, num_targets, initialize=True):
        super(Audio2Ml2Emo, self).__init__()

        self.preconcept_model = PreConceptModel()
        self.fc_ml = nn.Linear(256, 7)
        self.fc_ml2emo = nn.Linear(7, num_targets)

        # initialise weights
        if initialize:
            self.apply(initialize_weights)

    def get_last_layer_linear_weight(self):
        return self.fc_ml2emo.weight

    def forward(self, x):
        """ Forward pass. """
        x = self.preconcept_model(x)
        x = x.view(x.size(0), -1)
        ml = self.fc_ml(x)
        emo = self.fc_ml2emo(ml)
        return  {"output": emo, "c_prob": ml}


def initialize_weights(module):
    """ Function that initialises weights of given module. """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        if module.bias is not None:
            module.bias.data.zero_()
