#
# https://medium.com/@RobuRishabh/understanding-and-implementing-faster-r-cnn-248f7b25ff96
#

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Number of classes (your dataset classes + 1 for background)
num_classes = 3  # For example, 2 classes + background

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the head of the model with a new one (for the number of classes in your dataset)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define transformations (e.g., resizing, normalization)
transform = T.Compose([
    T.ToTensor(),
])
# Custom Dataset class or using an existing one
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        # Initialize dataset paths and annotations here
        self.transforms = transforms
        # Your dataset logic (image paths, annotations, etc.)

    def __getitem__(self, idx):
        # Load image
        img = ...  # Load your image here
        # Load corresponding bounding boxes and labels

        boxes = ...  # Load or define bounding boxes

        labels = ...  # Load or define labels

        # Create a target dictionary
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    def __len__(self):
        # Return the length of your dataset
        return len(self.data)

# Load dataset
dataset = CustomDataset(transforms=transform)
# Split into train and validation sets
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices[:-50])
valid_dataset = torch.utils.data.Subset(dataset, indices[-50:])
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                   collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False,
                                    collate_fn=lambda x: tuple(zip(*x)))

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
                                                   weight_decay=0.0005)
# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,
                                                               gamma=0.1)
# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

   # Training loop
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        losses.backward()
        optimizer.step()
        train_loss += losses.item()

    # Update the learning rate
    lr_scheduler.step()
    print(f'Epoch: {epoch + 1}, Loss: {train_loss / len(train_loader)}')
print("Training complete!")


# Set the model to evaluation mode
model.eval()
# Test on a new image
with torch.no_grad():
    for images, targets in valid_loader:
        images = list(img.to(device) for img in images)
        predictions = model(images)
        # Example: print the bounding boxes and labels for the first image
        print(predictions[0]['boxes'])
        print(predictions[0]['labels'])


import cv2
from PIL import Image
# Load image
img = Image.open("path/to/your/image.jpg")
# Apply the same transformation as for training
img = transform(img)
img = img.unsqueeze(0).to(device)
# Model prediction
model.eval()
with torch.no_grad():
    prediction = model([img])
# Print the predicted bounding boxes and labels
print(prediction[0]['boxes'])
print(prediction[0]['labels'])
