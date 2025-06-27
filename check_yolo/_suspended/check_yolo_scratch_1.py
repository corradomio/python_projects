#
# https://medium.com/@whyamit404/how-to-implement-a-yolo-object-detector-from-scratch-in-pytorch-e310829d92e6
#
import os
import cv2
import PIL.Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.optim as optim

import matplotlib.pyplot as plt

print("CUDA Available:", torch.cuda.is_available())



# class YOLO(nn.Module):
#     def __init__(self, num_classes=20, num_anchors=3, grid_size=7):
#         super(YOLO, self).__init__()
#         self.num_classes = num_classes
#         self.num_anchors = num_anchors
#         self.grid_size = grid_size
#
#         # Backbone: Feature extractor (e.g., simplified CNN for demonstration)
#         self.backbone = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#
#         # Detection Head: Outputs bounding boxes, confidence scores, and class probabilities
#         self.detector = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(256 * (grid_size // 4)**2, grid_size * grid_size * (num_anchors * 5 + num_classes)),
#         )
#
#     def forward(self, x):
#         features = self.backbone(x)
#         predictions = self.detector(features)
#         return predictions.view(-1, self.grid_size, self.grid_size, self.num_anchors * 5 + self.num_classes)
#
# # Instantiate the model
# model = YOLO(num_classes=20)
# print(model)

# ---

class ConvBlock(nn.Module):
    """A block of Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class YOLOBackbone(nn.Module):
    def __init__(self):
        super(YOLOBackbone, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layers(x)


class YOLOHead(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        super(YOLOHead, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()


class YOLO(nn.Module):
    def __init__(self, grid_size=7, num_classes=20, num_anchors=3):
        super(YOLO, self).__init__()
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(grid_size, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions

# Example usage
model = YOLO(grid_size=7, num_classes=20, num_anchors=3)
print(model)

# ---

def generate_anchors(scales, ratios):
    """Generates anchor boxes for given scales and aspect ratios."""
    anchors = []
    for scale in scales:
        for ratio in ratios:
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchors.append((width, height))
    return np.array(anchors)

# Example: Scales and ratios
scales = [0.1, 0.2, 0.4]
ratios = [0.5, 1, 2]
anchors = generate_anchors(scales, ratios)
print("Anchor Boxes:", anchors)

# ---

def convert_to_yolo_format(width, height, bbox):
    """Converts absolute bounding box to YOLO format."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return [x_center, y_center, box_width, box_height]

# ---

train_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# ---

def yolo_loss(predictions, targets, num_classes, lambda_coord=5, lambda_noobj=0.5):
    """
    Computes YOLO loss.
    - predictions: Predicted tensor.
    - targets: Ground truth tensor.
    """
    # Unpack predictions and targets
    pred_boxes = predictions[..., :4]
    pred_conf = predictions[..., 4]
    pred_classes = predictions[..., 5:]
    target_boxes = targets[..., :4]
    target_conf = targets[..., 4]
    target_classes = targets[..., 5:]

    # Localization Loss
    box_loss = lambda_coord * torch.sum((pred_boxes - target_boxes) ** 2)

    # Confidence Loss
    obj_loss = torch.sum((pred_conf - target_conf) ** 2)
    noobj_loss = lambda_noobj * torch.sum((pred_conf[target_conf == 0]) ** 2)

    # Classification Loss
    class_loss = torch.sum((pred_classes - target_classes) ** 2)

    # Total Loss
    total_loss = box_loss + obj_loss + noobj_loss + class_loss
    return total_loss

# ---

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))

        # Load image
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.open(img_path)

        # Load annotations
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_label, xc, yc, w, h = map(float, line.strip().split())
                boxes.append([class_label, xc, yc, w, h])

        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(boxes)

# Example: Initialize DataLoader
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

train_dataset = YOLODataset(
    img_dir=r"E:\Datasets\VOC2012\JPEGImages",
    label_dir=r"E:\Datasets\VOC2012\Annotations_yolo",
    # transforms=ToTensor()
    transforms=train_transforms
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ---

# Initialize model, optimizer, and loss function
model = YOLO(grid_size=7, num_classes=20, num_anchors=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = yolo_loss  # Your loss function from earlier

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        images = images.to('cuda')  # Move to GPU if available
        targets = targets.to('cuda')

        # Forward pass
        predictions = model(images)

        # Loss calculation
        loss = criterion(predictions, targets, num_classes=20)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# ---

def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union


def calculate_map(predictions, targets, num_classes, iou_threshold=0.5):
    """Calculates mean Average Precision."""
    average_precisions = []
    for c in range(num_classes):
        # Filter predictions and targets for class c
        pred_c = [p for p in predictions if p[0] == c]
        target_c = [t for t in targets if t[0] == c]

        # Sort predictions by confidence score
        pred_c.sort(key=lambda x: x[1], reverse=True)

        # Compute precision and recall
        precision = []
        recall = []
        for i, pred in enumerate(pred_c):
            iou = compute_iou(pred[2:], target_c[i][1:])
            if iou >= iou_threshold:
                precision.append(1)
                recall.append(1)
            else:
                precision.append(0)
                recall.append(0)

        # Calculate Average Precision
        average_precisions.append(sum(precision) / len(precision))
    return sum(average_precisions) / len(average_precisions)


def non_max_suppression(predictions, iou_threshold=0.5):
    """Applies NMS to suppress overlapping boxes."""
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    final_predictions = []

    while predictions:
        best = predictions.pop(0)
        predictions = [p for p in predictions if compute_iou(best[2:], p[2:]) < iou_threshold]
        final_predictions.append(best)

    return final_predictions


def perform_inference(model, image, conf_threshold=0.5):
    """Runs inference on an image and returns bounding boxes."""
    model.eval()
    with torch.no_grad():
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to('cuda')
        predictions = model(image_tensor)
        predictions = predictions.squeeze(0).cpu().numpy()

    # Filter boxes by confidence
    boxes = []
    for pred in predictions:
        if pred[4] >= conf_threshold:
            boxes.append(pred)

    # Apply Non-Maximum Suppression
    final_boxes = non_max_suppression(boxes)
    return final_boxes

# ---

def visualize_predictions(image, boxes, class_labels):
    """Draws bounding boxes on the image."""
    for box in boxes:
        class_label, conf, x, y, w, h = box
        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_labels[int(class_label)]} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# ---
