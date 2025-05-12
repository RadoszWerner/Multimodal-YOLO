from ultralytics import YOLO
import torch.nn as nn
import copy
import torch
from ultralytics.nn.modules import Concat, C2f, Conv, SPPF

# Ładowanie pretrenowanego modelu YOLOv8m
pretrained_model = YOLO('yolov8m.pt').model

# Ekstrakcja backbone'a (warstwy 0-9)
backbone = nn.Sequential(*list(pretrained_model.model.children())[:10])

# Definicja CustomBackbone
class CustomBackbone(nn.Module):
    def __init__(self, layers, out_idx=[4, 6, 8]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.out_idx = out_idx
        
    def forward(self, x):
        outputs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.out_idx:
                outputs.append(x)
        return outputs

# Tworzenie backbone'ów dla RGB i IR
backbone_rgb = CustomBackbone(backbone)
backbone_ir = copy.deepcopy(backbone_rgb)

# Modyfikacja pierwszej konwolucji dla IR (1 kanał)
backbone_ir.layers[0].conv = nn.Conv2d(1, 48, kernel_size=3, stride=2, padding=1, bias=False)

# Sprawdzenie kształtów wag (opcjonalnie)
print(backbone_rgb.layers[0].conv.weight.shape)  # [48, 3, 3, 3] dla RGB
print(backbone_ir.layers[0].conv.weight.shape)   # [48, 1, 3, 3] dla IR

# Definicja CustomNeck
class CustomNeck(nn.Module):
    def __init__(self, fused_channels):
        super().__init__()
        # fused_channels to liczba kanałów po konkatenacji, np. [192, 384, 576]
        self.layer9 = SPPF(fused_channels[2], fused_channels[2] // 2)  # SPPF dla najgłębszej skali
        self.layer10 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer11 = Concat()
        self.layer12 = C2f(fused_channels[2] // 2 + fused_channels[1], fused_channels[1] // 2, n=2)
        self.layer13 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer14 = Concat()
        self.layer15 = C2f(fused_channels[1] // 2 + fused_channels[0], fused_channels[0] // 2, n=2)
        self.layer16 = Conv(fused_channels[0] // 2, fused_channels[0] // 2, 3, s=2)
        self.layer17 = Concat()
        self.layer18 = C2f(fused_channels[0] // 2 + fused_channels[1], fused_channels[1] // 2, n=2)
        self.layer19 = Conv(fused_channels[1] // 2, fused_channels[1] // 2, 3, s=2)
        self.layer20 = Concat()
        self.layer21 = C2f(fused_channels[1] // 2 + fused_channels[2], fused_channels[2] // 2, n=2)

    def forward(self, fused):
        feat1, feat2, feat3 = fused  # feat1: warstwa 4, feat2: warstwa 6, feat3: warstwa 8

        x = self.layer9(feat3)
        x = self.layer10(x)
        x = self.layer11([x, feat2])
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14([x, feat1])
        feat_shallow = self.layer15(x)

        x = self.layer16(feat_shallow)
        x = self.layer17([x, feat2])
        feat_mid = self.layer18(x)
        x = self.layer19(feat_mid)
        x = self.layer20([x, feat3])
        feat_deep = self.layer21(x)

        return [feat_shallow, feat_mid, feat_deep]

# Definicja CustomYOLO
class CustomYOLO(nn.Module):
    def __init__(self, pretrained_model, backbone_rgb, backbone_ir):
        super().__init__()
        self.backbone_rgb = backbone_rgb
        self.backbone_ir = backbone_ir
        
        # Standardowe kanały dla YOLOv8m po warstwach 4, 6, 8 po konkatenacji: [192, 384, 576]
        self.neck_head = CustomNeck(fused_channels=[192, 384, 576])
        
        # Przeniesienie warstwy Detect z pretrenowanego modelu
        self.detect = copy.deepcopy(pretrained_model.model[-1])
        
    def forward(self, x_rgb, x_ir):
        # Ekstrakcja cech
        features_rgb = self.backbone_rgb(x_rgb)
        features_ir = self.backbone_ir(x_ir)
        
        # Fuzja przez konkatenację wzdłuż wymiaru kanałów
        fused = [torch.cat([f_rgb, f_ir], dim=1) for f_rgb, f_ir in zip(features_rgb, features_ir)]
        
        # Przetwarzanie przez neck
        neck_outputs = self.neck_head(fused)
        
        # Detekcja
        return self.detect(neck_outputs)

# Inicjalizacja modelu
custom_model = CustomYOLO(pretrained_model, backbone_rgb, backbone_ir)