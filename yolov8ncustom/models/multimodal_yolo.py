import torch
import copy
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights

class MultiModalYOLO(DetectionModel):
    def __init__(self, cfg="yolov8n.yaml"):
        super().__init__(cfg)
        
        # Clone backbones
        self.rgb_backbone = copy.deepcopy(self.model.backbone)
        self.ir_backbone = copy.deepcopy(self.model.backbone)
        
        # Modify IR backbone input channels
        self._modify_ir_backbone()
        
        # Fusion layers
        in_channels = self.rgb_backbone.out_channels
        self.fusion = torch.nn.ModuleList([
            torch.nn.Conv2d(2 * ch, ch, 1) for ch in in_channels
        ])
        
        # Keep original neck and head
        self.neck = self.model.neck
        self.head = self.model.head

    def _modify_ir_backbone(self):
        for name, module in self.ir_backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
                new_conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias is not None
                )
                initialize_weights(new_conv)
                
                # Replace layer
                *path, last = name.split('.')
                parent = self.ir_backbone
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, new_conv)
                break

    def forward(self, x, x_ir=None):
        if x_ir is None:
            x_ir = torch.zeros(x.size(0), 1, *x.shape[2:]).to(x.device)
        
        # Feature extraction
        rgb_features = self.rgb_backbone(x)
        ir_features = self.ir_backbone(x_ir)
        
        # Feature fusion
        fused = [
            self.fusion[i](torch.cat([r, i], dim=1))
            for i, (r, i) in enumerate(zip(rgb_features, ir_features))
        ]
        
        return self.head(self.neck(fused))