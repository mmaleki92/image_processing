import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SimpleYOLO(nn.Module):
    """
    A simplified YOLO model to detect basic shapes.
    This model is a lot simpler than full YOLO implementations
    but captures the essential concepts.
    """
    def __init__(self, num_classes=2, num_anchors=2):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Each grid cell predicts: [x, y, w, h, confidence, class1, class2]
        self.box_info_size = 5 + num_classes
        self.output_size = num_anchors * self.box_info_size
        
        # Feature extraction - Modified to ensure 13x13 output from 416x416 input
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 1/2
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 1/4
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 1/8
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 1/16
            
            # Add one more downsampling to get from 26x26 to 13x13
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 1/32
        )
        
        # Prediction
        self.prediction = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, self.output_size, kernel_size=1, stride=1, padding=0)
        )
        
        # Default anchor boxes (width, height) relative to grid cell size
        self.anchors = torch.tensor([
            [0.5, 0.5],  # Anchors for square-like objects
            [0.7, 0.7]   # Anchors for circle-like objects
        ])

    def forward(self, x):
        batch_size = x.size(0)
        features = self.features(x)
        output = self.prediction(features)
        
        # Reshape the output: [batch, num_anchors * box_info_size, grid_h, grid_w] -> 
        #                     [batch, grid_h, grid_w, num_anchors, box_info_size]
        grid_size = output.size(2)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, grid_size, grid_size, 
                            self.num_anchors, self.box_info_size)
        
        return output
class ShapesDataset(Dataset):
    """
    Generate synthetic dataset with circles and squares
    """
    def __init__(self, num_samples=1000, img_size=416, grid_size=13):
        self.num_samples = num_samples
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size // grid_size
        
        # Shape types: 0=square, 1=circle
        self.class_names = ["square", "circle"]
        self.num_classes = len(self.class_names)
        
        # Generate data
        self.images, self.targets = self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic images with shapes"""
        images = []
        targets = []
        
        for _ in range(self.num_samples):
            # Create a blank image with slight noise
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32) * 0.9
            img += np.random.randn(self.img_size, self.img_size, 3).astype(np.float32) * 0.1
            img = np.clip(img, 0, 1)
            
            # Number of shapes to draw (1-3)
            num_shapes = random.randint(1, 3)
            
            # Target tensor: [grid_h, grid_w, num_anchors, 5+num_classes]
            # 5 = [x, y, w, h, confidence]
            target = np.zeros((self.grid_size, self.grid_size, 2, 5+self.num_classes), dtype=np.float32)
            
            # Generate shapes
            for _ in range(num_shapes):
                # Random shape: 0=square, 1=circle
                shape_type = random.randint(0, self.num_classes-1)
                
                # Size between 10% and 30% of the image size
                size = random.randint(int(self.img_size*0.1), int(self.img_size*0.3))
                
                # Random position
                x_center = random.randint(size//2, self.img_size - size//2)
                y_center = random.randint(size//2, self.img_size - size//2)
                
                # Calculate bbox coordinates
                x_min = max(0, x_center - size//2)
                y_min = max(0, y_center - size//2)
                x_max = min(self.img_size, x_center + size//2)
                y_max = min(self.img_size, y_center + size//2)
                
                # Draw the shape
                if shape_type == 0:  # Square
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0.1, 0.2, 0.8), -1)
                else:  # Circle
                    radius = size // 2
                    cv2.circle(img, (x_center, y_center), radius, (0.8, 0.2, 0.1), -1)
                
                # Convert to grid cell coordinates
                grid_x = x_center // self.cell_size
                grid_y = y_center // self.cell_size
                
                # Width and height relative to cell size
                rel_width = (x_max - x_min) / self.cell_size
                rel_height = (y_max - y_min) / self.cell_size
                
                # Center position relative to grid cell
                rel_x = (x_center / self.cell_size) - grid_x
                rel_y = (y_center / self.cell_size) - grid_y
                
                # Choose best anchor based on IoU
                anchor_idx = 0  # Default to first anchor
                
                # Fill target tensor
                target[grid_y, grid_x, anchor_idx, 0] = rel_x
                target[grid_y, grid_x, anchor_idx, 1] = rel_y
                target[grid_y, grid_x, anchor_idx, 2] = rel_width
                target[grid_y, grid_x, anchor_idx, 3] = rel_height
                target[grid_y, grid_x, anchor_idx, 4] = 1.0  # confidence
                target[grid_y, grid_x, anchor_idx, 5 + shape_type] = 1.0  # class
            
            images.append(img)
            targets.append(target)
            
        return images, targets
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return an image and its target"""
        img = self.images[idx]
        target = self.targets[idx]
        
        # Convert to torch tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        target_tensor = torch.from_numpy(target)
        
        return img_tensor, target_tensor
    
    def visualize(self, idx, save_path=None):
        """Visualize an image with its bounding boxes"""
        img = self.images[idx]
        target = self.targets[idx]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        
        # Draw bounding boxes
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for a in range(2):  # 2 anchors
                    confidence = target[i, j, a, 4]
                    if confidence > 0:
                        # Extract box coordinates
                        rel_x = target[i, j, a, 0]
                        rel_y = target[i, j, a, 1]
                        rel_w = target[i, j, a, 2]
                        rel_h = target[i, j, a, 3]
                        
                        # Convert to image coordinates
                        x_center = (j + rel_x) * self.cell_size
                        y_center = (i + rel_y) * self.cell_size
                        width = rel_w * self.cell_size
                        height = rel_h * self.cell_size
                        
                        # Calculate box corners
                        x_min = x_center - width/2
                        y_min = y_center - height/2
                        x_max = x_center + width/2
                        y_max = y_center + height/2
                        
                        # Determine class
                        class_scores = target[i, j, a, 5:]
                        class_id = np.argmax(class_scores)
                        class_name = self.class_names[class_id]
                        
                        # Draw rectangle
                        rect = plt.Rectangle((x_min, y_min), width, height, 
                                          linewidth=2, edgecolor='r', facecolor='none')
                        plt.gca().add_patch(rect)
                        plt.text(x_min, y_min, class_name, 
                                 bbox=dict(facecolor='red', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def intersection_over_union(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1: tensor of shape (N, 4) representing bounding boxes
               format: (x_min, y_min, x_max, y_max)
        boxes2: tensor of shape (M, 4) representing bounding boxes
               format: (x_min, y_min, x_max, y_max)
    
    Returns:
        iou: tensor of shape (N, M) containing IoU values for each pair of boxes
    """
    # Get box dimensions
    x1_min, y1_min, x1_max, y1_max = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x2_min, y2_min, x2_max, y2_max = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    # Calculate intersection dimensions
    x_intersection_min = torch.max(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    y_intersection_min = torch.max(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    x_intersection_max = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    y_intersection_max = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))
    
    # Calculate intersection area
    width = torch.clamp(x_intersection_max - x_intersection_min, min=0)
    height = torch.clamp(y_intersection_max - y_intersection_min, min=0)
    intersection_area = width * height
    
    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = (area1.unsqueeze(1) + area2.unsqueeze(0)) - intersection_area
    
    # Calculate IoU
    iou = intersection_area / (union_area + 1e-6)
    return iou

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression to remove overlapping bounding boxes
    
    Args:
        boxes: tensor of shape (N, 4) representing bounding boxes
               format: (x_min, y_min, x_max, y_max)
        scores: tensor of shape (N) representing confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep_indices: indices of boxes to keep
    """
    # Sort boxes by confidence score
    _, sorted_indices = torch.sort(scores, descending=True)
    
    keep_indices = []
    while sorted_indices.size(0) > 0:
        # Pick the box with highest confidence score
        current_index = sorted_indices[0]
        keep_indices.append(current_index)
        
        # Break if there's only one box left
        if sorted_indices.size(0) == 1:
            break
            
        # Get remaining boxes
        remaining_indices = sorted_indices[1:]
        
        # Get IoU of the current box with all remaining boxes
        current_box = boxes[current_index].unsqueeze(0)
        remaining_boxes = boxes[remaining_indices]
        
        ious = intersection_over_union(current_box, remaining_boxes).squeeze()
        
        # Keep boxes with IoU less than threshold
        mask = ious < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    return torch.tensor(keep_indices)

def process_predictions(predictions, anchors, img_size=416, grid_size=13, 
                       confidence_threshold=0.5, iou_threshold=0.5):
    """
    Process YOLO model predictions to get final bounding boxes
    
    Args:
        predictions: model output tensor of shape [batch, grid_h, grid_w, anchors, 5+num_classes]
        anchors: anchor boxes
        img_size: size of the input image
        grid_size: size of the grid
        confidence_threshold: threshold for confidence scores
        iou_threshold: threshold for NMS
    
    Returns:
        boxes: list of bounding boxes (x_min, y_min, x_max, y_max)
        scores: list of confidence scores
        classes: list of class indices
    """
    batch_size = predictions.size(0)
    cell_size = img_size // grid_size
    num_anchors = anchors.size(0)
    num_classes = predictions.size(-1) - 5
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for b in range(batch_size):
        batch_boxes = []
        batch_scores = []
        batch_classes = []
        
        # For each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                # For each anchor
                for a in range(num_anchors):
                    # Get confidence score
                    confidence = predictions[b, i, j, a, 4]
                    
                    if confidence > confidence_threshold:
                        # Get box coordinates
                        rel_x = predictions[b, i, j, a, 0]
                        rel_y = predictions[b, i, j, a, 1]
                        rel_w = predictions[b, i, j, a, 2]
                        rel_h = predictions[b, i, j, a, 3]
                        
                        # Convert to absolute coordinates
                        x_center = (j + rel_x) * cell_size
                        y_center = (i + rel_y) * cell_size
                        width = rel_w * cell_size
                        height = rel_h * cell_size
                        
                        # Convert to corner format
                        x_min = x_center - width/2
                        y_min = y_center - height/2
                        x_max = x_center + width/2
                        y_max = y_center + height/2
                        
                        # Get class scores
                        class_scores = predictions[b, i, j, a, 5:5+num_classes]
                        class_id = torch.argmax(class_scores).item()
                        class_score = class_scores[class_id]
                        
                        # Final score combines confidence and class score
                        score = confidence * class_score
                        
                        batch_boxes.append([x_min, y_min, x_max, y_max])
                        batch_scores.append(score)
                        batch_classes.append(class_id)
        
        # Apply NMS
        if batch_boxes:
            batch_boxes = torch.tensor(batch_boxes)
            batch_scores = torch.tensor(batch_scores)
            batch_classes = torch.tensor(batch_classes)
            
            keep_indices = non_max_suppression(batch_boxes, batch_scores, iou_threshold)
            
            all_boxes.append(batch_boxes[keep_indices])
            all_scores.append(batch_scores[keep_indices])
            all_classes.append(batch_classes[keep_indices])
        else:
            all_boxes.append(torch.tensor([]))
            all_scores.append(torch.tensor([]))
            all_classes.append(torch.tensor([]))
    
    return all_boxes, all_scores, all_classes
class YOLOLoss(torch.nn.Module):
    """
    Loss function for YOLO training
    """
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="sum")
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, predictions, targets):
        """
        Calculate the YOLO loss
        
        Args:
            predictions: model output tensor of shape [batch, grid_h, grid_w, anchors, 5+num_classes]
            targets: ground truth tensor of same shape
        
        Returns:
            loss: scalar tensor representing the total loss
        """
        batch_size = predictions.size(0)
        
        # Extract components
        pred_xy = predictions[..., 0:2]
        pred_wh = predictions[..., 2:4]
        pred_conf = predictions[..., 4:5]
        pred_class = predictions[..., 5:]
        
        target_xy = targets[..., 0:2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4:5]
        target_class = targets[..., 5:]
        
        # Create masks
        obj_mask = target_conf > 0.5  # Where objects exist
        noobj_mask = ~obj_mask
        
        # Expand obj_mask for coordinate losses to match dimensions
        # This creates a mask that can be properly broadcast to the xy and wh tensors
        obj_mask_expanded = obj_mask.expand_as(pred_xy[..., 0:1]).repeat(1, 1, 1, 1, 2)
        
        # Calculate losses
        
        # Coordinate loss (only for cells that contain objects)
        # Use the expanded mask for the coordinate tensors
        xy_loss = self.mse(pred_xy[obj_mask_expanded], target_xy[obj_mask_expanded])
        wh_loss = self.mse(pred_wh[obj_mask_expanded], target_wh[obj_mask_expanded])
        coord_loss = self.lambda_coord * (xy_loss + wh_loss)
        
        # Confidence loss - this uses the original mask that matches the dimensions
        conf_obj_loss = self.mse(pred_conf[obj_mask], target_conf[obj_mask])
        conf_noobj_loss = self.lambda_noobj * self.mse(pred_conf[noobj_mask], target_conf[noobj_mask])
        conf_loss = conf_obj_loss + conf_noobj_loss
        
        # Class loss - expand the mask for class predictions which might have multiple classes
        obj_mask_class = obj_mask.expand_as(pred_class[..., 0:1]).repeat(1, 1, 1, 1, pred_class.size(-1))
        class_loss = self.bce(pred_class[obj_mask_class], target_class[obj_mask_class])
        
        # Total loss
        total_loss = (coord_loss + conf_loss + class_loss) / batch_size
        
        return total_loss


# Configuration
IMG_SIZE = 416
GRID_SIZE = 13
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_SAMPLES = 500
NUM_CLASSES = 2
SAVE_DIR = "checkpoints"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create dataset and dataloader
dataset = ShapesDataset(num_samples=NUM_SAMPLES, img_size=IMG_SIZE, grid_size=GRID_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Visualize some examples
print("Visualizing examples...")
for i in range(3):
    dataset.visualize(i, save_path=f"example_{i}.png")

# Initialize model and loss
model = SimpleYOLO(num_classes=NUM_CLASSES).to(device)
criterion = YOLOLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("Starting training...")
losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    # Average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
    
    # Save model checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth"))

# Save final model
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, os.path.join(SAVE_DIR, "model_final.pth"))

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()

print("Training completed!")