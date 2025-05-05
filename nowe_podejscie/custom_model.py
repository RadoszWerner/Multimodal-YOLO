from ultralytics import YOLO
import torch
import torch.nn as nn
from copy import deepcopy
import yaml
import os
from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, Detect

class DualBackboneYOLO(YOLO):
    def __init__(self, model='custom_yolov8_dual_backbone.yaml', pretrained=True):
        # Wczytaj niestandardowy plik YAML
        with open(model, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        # Inicjalizuj nadrzędną klasę YOLO bez modelu, ustawiamy task='detect'
        super().__init__(model='yolov8m.yaml', task='detect')  # Używamy domyślnego YAML jako placeholder

        # Zainicjuj niestandardowy model
        self.model = CustomDualBackboneModel(cfg_dict, nc=cfg_dict['nc'])
        self.cfg_dict = cfg_dict  # Zapisz konfigurację dla innych metod

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        # Ładuj pretrenowane wagi YOLOv8m dla backbone_rgb
        pretrained_model = YOLO("yolov8m.pt")
        state_dict = pretrained_model.model.state_dict()

        # Kopiuj wagi do backbone_rgb
        rgb_state_dict = {k: v for k, v in state_dict.items() if k.startswith("model.")}  # Wszystkie warstwy
        model_state_dict = self.model.state_dict()
        rgb_state_dict = {k: v for k, v in rgb_state_dict.items() if k in model_state_dict}
        model_state_dict.update(rgb_state_dict)

        # Inicjalizuj backbone_ir, kopiując wagi z RGB i dostosowując pierwszą warstwę
        ir_state_dict = deepcopy(rgb_state_dict)
        first_conv_key = "backbone_ir.0.conv.weight"  # Pierwsza warstwa backbone_ir
        rgb_weight_key = "backbone_rgb.0.conv.weight"  # Pierwsza warstwa backbone_rgb
        if first_conv_key in model_state_dict and rgb_weight_key in rgb_state_dict:
            rgb_weight = rgb_state_dict[rgb_weight_key]  # [64, 3, 3, 3]
            ir_weight = rgb_weight.mean(dim=1, keepdim=True)  # Średnia po kanałach: [64, 1, 3, 3]
            ir_state_dict[first_conv_key] = ir_weight
            model_state_dict.update(ir_state_dict)

        self.model.load_state_dict(model_state_dict)

    def forward(self, x):
        rgb, ir = x["rgb"], x["ir"]  # Oczekuj słownika z dwoma wejściami
        return self.model([rgb, ir])  # Przekaż jako lista do modelu

class CustomDualBackboneModel(nn.Module):
    def __init__(self, cfg, nc=1, verbose=True):
        super().__init__()
        self.yaml = cfg
        self.nc = nc
        self.verbose = verbose

        # Zbuduj model na podstawie niestandardowej konfiguracji
        self.backbone_rgb, self.backbone_ir, self.neck, self.head = self.build_custom_model()

    def build_custom_model(self):
        backbone_rgb = []
        backbone_ir = []
        neck = []
        head = []

        # Backbone RGB
        for layer_cfg in self.yaml['backbone_rgb']:
            layer_type = layer_cfg[2]
            args = layer_cfg[3]
            print(f"Budowanie warstwy RGB: {layer_type}, args: {args}")  # Debugowanie
            if layer_type == 'Conv':
                backbone_rgb.append(Conv(*args))
            elif layer_type == 'C2f':
                backbone_rgb.append(C2f(*args))
            elif layer_type == 'SPPF':
                backbone_rgb.append(SPPF(*args))

        # Backbone IR
        for layer_cfg in self.yaml['backbone_ir']:
            layer_type = layer_cfg[2]
            args = layer_cfg[3]
            print(f"Budowanie warstwy IR: {layer_type}, args: {args}")  # Debugowanie
            if layer_type == 'Conv':
                backbone_ir.append(Conv(*args))
            elif layer_type == 'C2f':
                backbone_ir.append(C2f(*args))
            elif layer_type == 'SPPF':
                backbone_ir.append(SPPF(*args))

        # Neck
        for layer_cfg in self.yaml['neck']:
            layer_type = layer_cfg[2]
            args = layer_cfg[3]
            print(f"Budowanie warstwy Neck: {layer_type}, args: {args}")  # Debugowanie
            if layer_type == 'Conv':
                neck.append(Conv(*args))
            elif layer_type == 'Upsample':
                neck.append(nn.Upsample(*args))
            elif layer_type == 'Concat':
                neck.append(Concat(*args))
            elif layer_type == 'C2f':
                neck.append(C2f(*args))

        # Head
        for layer_cfg in self.yaml['head']:
            layer_type = layer_cfg[2]
            args = layer_cfg[3]
            print(f"Budowanie warstwy Head: {layer_type}, args: {args}, nc: {self.nc}")  # Debugowanie
            if layer_type == 'Conv':
                head.append(Conv(*args))
            elif layer_type == 'Detect':
                # Zamień 'nc' w args na self.nc (dla elastyczności)
                if isinstance(args, list) and 'nc' in args:
                    args = [self.nc if x == 'nc' else x for x in args]
                print(f"Po zamianie args dla Detect: {args}")  # Dodatkowe debugowanie
                head.append(Detect(self.nc, args))  # Przekazuj args jako listę, bez rozpakowywania

        return (
            nn.Sequential(*backbone_rgb),
            nn.Sequential(*backbone_ir),
            nn.Sequential(*neck),
            nn.Sequential(*head)
        )

    def forward(self, x):
        rgb, ir = x[0], x[1]

        # Przetwarzanie przez backbone_rgb
        rgb_features = self.backbone_rgb(rgb)

        # Przetwarzanie przez backbone_ir
        ir_features = self.backbone_ir(ir)

        # Feature fusion (Concat)
        x = self.neck[0]([rgb_features, ir_features])  # Pierwszy Concat w neck

        # Przetwarzanie przez resztę neck
        for layer in self.neck[1:]:
            if isinstance(layer, Concat):
                # Zakładamy, że Concat używa wcześniejszych warstw (np. z backbone_rgb[6], backbone_rgb[4])
                x = layer([x, rgb_features if layer.from_idx[1] == 6 else ir_features])
            else:
                x = layer(x)

        # Przetwarzanie przez head
        x = self.head(x)

        return x

    def _load_state_dict(self, state_dict):
        """Metoda pomocnicza do ładowania wag, zgodna z ultralytics."""
        self.load_state_dict(state_dict)