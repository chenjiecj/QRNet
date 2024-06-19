
import torch.nn.functional as F
from torch import nn

from .ops import NestedTensor, nested_tensor_from_tensor_list, unused
from .decode import Decoder

from .resnet import ResNet
from .backbone import SFMBackbone
from nets.efficientnet import EfficientNet as EffNet
from .mobilenet import mobilenet_v1

class MobileNetV1(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV1, self).__init__()
        self.model = mobilenet_v1(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.stage1(x)
        out4 = self.model.stage2(out3)
        out5 = self.model.stage3(out4)
        return out3, out4, out5
class EfficientNet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', pretrained)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model
    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2] :
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[2:]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class QRNet(nn.Module):
    def __init__(self, backbone, num_classes, aux_loss= True, inference_mode =False):
        super().__init__()
        self.inference_mode = inference_mode
        if backbone== "resnet50":
            self.backbone = ResNet(depth=50)
            self.decoder = Decoder(num_classes=num_classes, hidden_dim=128,
                                                 feat_channels=[512, 1024, 2048], num_decoder_layers=2,
                                                 dim_feedforward=512, aux_loss=aux_loss)
        elif backbone == "mobilenet":
            self.backbone = MobileNetV1()
            self.decoder = Decoder(num_classes=num_classes, hidden_dim=128,
                                                 feat_channels=[256, 512, 1024], num_decoder_layers=2,
                                                 dim_feedforward=512, aux_loss=aux_loss)
        elif backbone == "qrnet":
            self.backbone = SFMBackbone()
            self.decoder = Decoder(num_classes=num_classes, hidden_dim=128,
                                                 feat_channels=[64, 128, 256], num_decoder_layers=2, dim_feedforward=512, aux_loss=aux_loss)
        elif backbone == "efficientnet":
            self.backbone = EfficientNet(0)
            self.decoder = Decoder(num_classes=num_classes, hidden_dim=128,
                                                 feat_channels=[40, 112, 320 ], num_decoder_layers=2, dim_feedforward=512, aux_loss=aux_loss)
        else:
            print("backbone format err")
    def forward(self, x, targets=None):
        x      = self.backbone(x)
        x      = self.decoder(x, targets)
        return x

    @unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



