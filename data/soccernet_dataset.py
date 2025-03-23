import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class SoccerNetTrackingDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Match images with their corresponding label files
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(".jpg")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_filename)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x1, y1, x2, y2 = map(float, parts[1:])
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # target = {
        #     "boxes": boxes,
        #     "labels": labels,
        #     "image_id": torch.tensor([idx])
        # }

        # Apply image transforms if given
        if self.transform:
            image = self.transform(image)

        return image, boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))