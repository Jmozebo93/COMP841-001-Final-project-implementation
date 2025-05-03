import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from mobileNetv3_scratch import MobileNetV3, Swish, SEBlock, MBConv
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from mobileNetv3_scratch import MobileNetV3, Swish
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import time
from torch.amp import autocast, GradScaler  # Mixed Precision imports
import shutil

# Paths
data_dir = './data_converted'  # Directory containing the dataset

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_sizes = 32
epochs_options = 50
learning_rate = 0.001
num_classes = 2

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Augmentation
    transforms.RandomRotation(10), # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Dataset split ratios
train_ratio = 0.75  # 75% for training
val_ratio = 0.25    # 25% for validation

# Calculate split sizes
total_size = len(full_dataset)
train_val_size = int(train_ratio * total_size) # 75% of the dataset
test_size = total_size - train_val_size  # Remaining  25% for testing

# Split into train+val and test
train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size])

# Further split train+val into train and val
train_ratio = 0.8 # 80% of train+val for training
val_ratio = 0.2  # 20% of train+val for validation

train_size = int(train_ratio * train_val_size)  # 80% of train+val for training
val_size = train_val_size - train_size  # Remaining 20% for validation

train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")



# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=False, num_workers=4, pin_memory=True)
test_dataset = datasets.ImageFolder(root='./test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, num_workers=4, pin_memory=True)

# Mixed Precision Setup
scaler = GradScaler()

# Training function
def train_model(batch_size, num_epochs):

    # Load MobileNetV3 model
    model = MobileNetV3(num_classes=num_classes)
    model.classifier = nn.Sequential(
        nn.Linear(160, 1280),  # Bottleneck layer
        Swish(),
        nn.Linear(1280, num_classes)  # Final classification layer
    )
    
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    

    print(f"Training with batch size {batch_size} for {num_epochs} epochs")
    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Mixed Precision: Autocast and gradient scaling
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        print(f"Epoch time: {time.time() - start_time:.2f} seconds")

    # Validation Accuracy
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:,1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels for classification report
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate validation accuracy
    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy with batch size {batch_size}: {val_accuracy:.2f}%')

    # Step the learning rate scheduler
    scheduler.step(val_accuracy)

    torch.save(model.state_dict(), './saved_models/mobileNetv3_sickle_cell.pth')
    print("Model saved.")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    def plot_confusion_matrix(conf_matrix, class_names):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    # Call the plot function
    plot_confusion_matrix(conf_matrix, full_dataset.classes)

    # Calculate ROC and AUC 
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc:.2f}')

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# Run experiments
train_model(batch_sizes, epochs_options)
