import pytorch_lightning as pl
import torchvision.models as models
import torch
import torch.nn.functional as F
from src.utils import read_yaml_config_file
import torch.nn as nn


class PretrainedDenseNet(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.feature_extractor = models.densenet121(pretrained=True)
        self.feature_extractor.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        n_sizes = self._get_conv_output(self.input_shape)

        self.classifier = nn.Linear(n_sizes, self.num_classes)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x), dim=1)
        # x = F.softmax(self.classifier(x), dim=1)
        return x
