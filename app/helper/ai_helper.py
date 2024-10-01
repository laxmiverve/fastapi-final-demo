import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# Parse COCO Annotations
def parseCocoAnnotations(annotation_file, base_path):
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        images = annotations['images']
        anns = annotations['annotations']
        
        image_paths = []
        labels = []
        for ann in anns:
            image_id = ann['image_id']
            image_info = next(img for img in images if img['id'] == image_id)
            image_paths.append(os.path.join(base_path, image_info['file_name']))
            labels.append(ann['category_id'])  # Assuming 'category_id' is zero-indexed
        
        return image_paths, labels, annotations['categories']  # Return categories for mapping
    except Exception as e:
        print("An exception occurred during coco annotation:", str(e))


# Create category mapping and the number of classes
def createCategoryMapping(categories):
    try:
        category_mapping = {cat['id']: cat['name'] for cat in categories}
        return category_mapping, len(category_mapping)
    except Exception as e:
        print("An exception occurred:", str(e))


# Dataset paths
dataset_path = 'datasets/image-classify.v1i.coco'
train_image_paths, train_labels, train_categories = parseCocoAnnotations(
    os.path.join(dataset_path, 'train', '_annotations.coco.json'), 
    os.path.join(dataset_path, 'train')
)
valid_image_paths, valid_labels, _ = parseCocoAnnotations(
    os.path.join(dataset_path, 'valid', '_annotations.coco.json'), 
    os.path.join(dataset_path, 'valid')
)
test_image_paths, test_labels, _ = parseCocoAnnotations(
    os.path.join(dataset_path, 'test', '_annotations.coco.json'), 
    os.path.join(dataset_path, 'test')
)

# Create category mapping
category_mapping, num_classes = createCategoryMapping(train_categories)


# Custom Dataset for PyTorch
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define Transforms (Preprocess: Resize, Normalize, Augment)
IMG_SIZE = (224, 224)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225]),
])

train_dataset = ImageClassificationDataset(train_image_paths, train_labels, transform = transform)
valid_dataset = ImageClassificationDataset(valid_image_paths, valid_labels, transform = transform)
test_dataset = ImageClassificationDataset(test_image_paths, test_labels, transform = transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 4)
valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers = 4)

# Define MobileNet V2 Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def createMobilenetV2(num_classes):
    try:
        model = models.mobilenet_v2(weights = 'DEFAULT')  # Use the new weights argument
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    except Exception as e:
        print("An exception occurred during load model:", str(e))


model = createMobilenetV2(num_classes).to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

# Train and Validate the Model
def trainModel(model, train_loader, valid_loader, criterion, optimizer, num_epochs = 10):
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
            
            validate(model, valid_loader, criterion)
    except Exception as e:
        print("An exception occurred during model training:", str(e))


def validate(model, valid_loader, criterion):
    try:
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f'Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}')
    except Exception as e:
        print("An exception occurred during validate model:", str(e))


# Train the model
trainModel(model, train_loader, valid_loader, criterion, optimizer, num_epochs = 10)

# Test the Model
def testModel(model, test_loader):
    try:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
    except Exception as e:
        print("An exception occurred during test the model:", str(e))


testModel(model, test_loader)

# Save the model for future use 
torch.save(model.state_dict(), 'mobilenet_v2_image_classifier.pth')




'''
This model is trained using these specific images, including screenshots from
Chrome, Gmail, Skype, Files, Terminal, VIS, VS Code 
'''