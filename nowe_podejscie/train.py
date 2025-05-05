from torch.utils.data import DataLoader
from custom_dataset import MultimodalYOLODataset, custom_collate_fn
from custom_model import DualBackboneYOLO

# Utworzenie datasetów
train_dataset = MultimodalYOLODataset(
    data_dir='D:/Projects/multimodal-YOLO/llvip_mod/train',
    rgb_subdir='rgb',
    ir_subdir='ir',
    annotations_dir='Annotations',
    img_size=(640, 640),
    augment=True
)

val_dataset = MultimodalYOLODataset(
    data_dir='D:/Projects/multimodal-YOLO/llvip_mod/val',
    rgb_subdir='rgb',
    ir_subdir='ir',
    annotations_dir='Annotations',
    img_size=(640, 640),
    augment=False
)

# Utworzenie dataloaderów
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=custom_collate_fn
)

# Inicjalizacja modelu
model = DualBackboneYOLO(model='custom_yolov8_dual_backbone.yaml', pretrained=True)

# Trenowanie
model.train(
    data='data.yaml',
    epochs=1,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    optimizer='AdamW',
    lr0=0.001,
    patience=50,
    save_period=10,
    project='runs/train',
    name='dual_backbone_exp'
)