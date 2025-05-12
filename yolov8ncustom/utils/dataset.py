from ultralytics.data.dataset import YOLODataset
import cv2
import os

class MultimodalDataset(YOLODataset):
    def __init__(self, *args, streams=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.streams = streams or ['rgb', 'ir']
        
        # Get IR image paths
        self.ir_im_files = [
            os.path.join(self.streams['ir'], os.path.basename(p))
            for p in self.im_files
        ]

    def load_image(self, index):
        # Load RGB
        rgb_img = cv2.cvtColor(cv2.imread(self.im_files[index]), cv2.COLOR_BGR2RGB)
        
        # Load IR
        ir_img = cv2.imread(self.ir_im_files[index], cv2.IMREAD_GRAYSCALE)
        
        return {'rgb': rgb_img, 'ir': ir_img}, self.labels[index]

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch['ir'] = self._load_ir_image(index)
        return batch

    def _load_ir_image(self, index):
        ir_path = self.ir_im_files[index]
        img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel
        return self._apply_transforms(img)