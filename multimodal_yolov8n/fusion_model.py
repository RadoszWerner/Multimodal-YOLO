import torch
import torch.nn as nn
from ultralytics import YOLO

class FusionYOLO(nn.Module):
    def __init__(self, model_rgb, model_ir):
        super(FusionYOLO, self).__init__()
        # Backbone dla RGB
        self.backbone_rgb = model_rgb.model.model[:10]  # Do warstwy 10 (backbone)
        
        # Backbone dla IR z zmodyfikowaną pierwszą warstwą
        self.backbone_ir = model_ir.model.model[:10]
        conv_layer = self.backbone_ir[0].conv
        new_conv = nn.Conv2d(1, conv_layer.out_channels, kernel_size=conv_layer.kernel_size,
                            stride=conv_layer.stride, padding=conv_layer.padding,
                            bias=conv_layer.bias is not None)
        with torch.no_grad():
            new_conv.weight.data = conv_layer.weight.data.mean(dim=1, keepdim=True)
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.data
        self.backbone_ir[0].conv = new_conv

        # Warstwa fuzji (konkatenacja + konwolucja 1x1)
        self.fusion_layer = nn.Conv2d(512, 256, kernel_size=1)  # Dostosuj wymiary
        
        # Szyja i głowa
        self.neck_head = model_rgb.model.model[10:]

    def forward(self, x_rgb, x_ir):
        # Przetwarzanie przez backbone
        feat_rgb = self.backbone_rgb(x_rgb)
        feat_ir = self.backbone_ir(x_ir)
        
        # Fuzja cech (konkatenacja ostatniej warstwy backbone)
        fused = torch.cat((feat_rgb[-1], feat_ir[-1]), dim=1)
        fused = self.fusion_layer(fused)
        
        # Przekazanie do szyi i głowy
        out = self.neck_head([fused])
        return out

# Funkcja do ładowania modelu
def load_fusion_model():
    model_rgb = YOLO('yolov8n.pt')
    model_ir = YOLO('yolov8n.pt')
    return FusionYOLO(model_rgb, model_ir)