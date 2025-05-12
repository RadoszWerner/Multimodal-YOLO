import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.data.dataset import YOLODataset
from fusion_model import load_fusion_model

class FusionDataset(Dataset):
    def __init__(self, data_rgb, data_ir):
        self.dataset_rgb = YOLODataset(data=data_rgb, imgsz=640)
        self.dataset_ir = YOLODataset(data=data_ir, imgsz=640)
        assert len(self.dataset_rgb) == len(self.dataset_ir), "Dataset sizes do not match"

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, idx):
        img_rgb, label_rgb = self.dataset_rgb[idx]
        img_ir, _ = self.dataset_ir[idx]  # Zakładamy te same etykiety dla RGB i IR
        return img_rgb, img_ir, label_rgb

def train():
    # Konfiguracja
    data_rgb = 'configs/data_rgb.yaml'  # Dane RGB
    data_ir = 'configs/data_ir.yaml'  # Dane IR (zmień ścieżki w data.yaml dla IR)
    batch_size = 16
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset i DataLoader
    dataset = FusionDataset(data_rgb, data_ir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = load_fusion_model().to(device)

    # Optymalizator i funkcja straty
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Funkcja straty YOLO (zaadaptuj z Ultralytics)

    for epoch in range(epochs):
        for batch in dataloader:
            img_rgb, img_ir, labels = batch
            img_rgb, img_ir, labels = img_rgb.to(device), img_ir.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img_rgb, img_ir)
            loss = compute_loss(outputs, labels)  # Zaimplementuj funkcję straty
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()