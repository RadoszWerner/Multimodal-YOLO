from torch.utils.data import Dataset, DataLoader, default_collate
import cv2
import os
import torch

class MultimodalYOLODataset(Dataset):
    def __init__(self, data_dir, rgb_subdir='rgb', ir_subdir='ir', annotations_dir='Annotations', img_size=(640, 640), augment=False):
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, rgb_subdir)
        self.ir_dir = os.path.join(data_dir, ir_subdir)
        self.annotations_dir = os.path.join(data_dir, '..', annotations_dir)
        self.img_files = sorted(os.listdir(self.rgb_dir))
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        rgb_path = os.path.join(self.rgb_dir, img_name)
        ir_path = os.path.join(self.ir_dir, img_name)
        label_path = os.path.join(self.annotations_dir, img_name.replace('.jpg', '.txt'))

        # Wczytanie obrazów
        img_rgb = cv2.imread(rgb_path)
        if img_rgb is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu RGB: {rgb_path}")
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, self.img_size)

        img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if img_ir is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu IR: {ir_path}")
        img_ir = cv2.resize(img_ir, self.img_size)
        img_ir = img_ir[..., None]  # (H, W) -> (H, W, 1)

        # Augmentacje (opcjonalne)
        if self.augment:
            img_rgb, img_ir = self.apply_augmentations(img_rgb, img_ir)

        # Konwersja na tensory i normalizacja
        img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # [3, 640, 640]
        img_ir = torch.from_numpy(img_ir).permute(2, 0, 1).float() / 255.0   # [1, 640, 640]

        # Wczytanie adnotacji
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [list(map(float, line.split())) for line in f.readlines()]
            labels = torch.tensor(labels) if labels else torch.zeros((0, 5))
            if labels.numel() > 0:
                labels = torch.cat([torch.zeros((labels.shape[0], 1)), labels], dim=1)  # Dodaj batch_idx
            else:
                labels = torch.zeros((0, 6))  # Pusty tensor [batch_idx, class, x, y, w, h]
        else:
            labels = torch.zeros((0, 6))

        return {"rgb": img_rgb, "ir": img_ir, "label": labels}

    def apply_augmentations(self, img_rgb, img_ir):
        # Opcjonalne augmentacje (np. flip, rotation)
        return img_rgb, img_ir

def custom_collate_fn(batch):
    rgb_imgs = [item["rgb"] for item in batch]
    ir_imgs = [item["ir"] for item in batch]
    targets = [item["label"] for item in batch]

    rgb_imgs = default_collate(rgb_imgs)  # [batch, 3, 640, 640]
    ir_imgs = default_collate(ir_imgs)    # [batch, 1, 640, 640]

    # Dodaj batch_idx do etykiet
    for i, target in enumerate(targets):
        if target.numel() > 0:
            target[:, 0] = i  # Ustaw batch_idx
    targets = [t for t in targets if t.numel() > 0]  # Usuń puste etykiety
    if targets:
        targets = torch.cat(targets, dim=0)  # Połącz w jeden tensor
    else:
        targets = torch.zeros((0, 6))

    return {"rgb": rgb_imgs, "ir": ir_imgs, "label": targets}