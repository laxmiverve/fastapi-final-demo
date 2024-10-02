import os
import json
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

class ImageClassifier:
    def __init__(self, dataset_path, num_epochs = 1, batch_size = 32, lr = 0.0001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = dataset_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.train_loader, self.valid_loader, self.test_loader, self.num_classes = self.loadData()
        self.model = self.createModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

    # Load Data
    def loadData(self):
        train_image_paths, train_labels, train_categories = self.parseCocoAnnotations(
            os.path.join(self.dataset_path, 'train', '_annotations.coco.json'), 
            os.path.join(self.dataset_path, 'train')
        )
        valid_image_paths, valid_labels, _ = self.parseCocoAnnotations(
            os.path.join(self.dataset_path, 'valid', '_annotations.coco.json'), 
            os.path.join(self.dataset_path, 'valid')
        )
        test_image_paths, test_labels, _ = self.parseCocoAnnotations(
            os.path.join(self.dataset_path, 'test', '_annotations.coco.json'), 
            os.path.join(self.dataset_path, 'test')
        )

        category_mapping, num_classes = self.createCategoryMapping(train_categories)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = ImageClassificationDataset(train_image_paths, train_labels, transform=transform)
        valid_dataset = ImageClassificationDataset(valid_image_paths, valid_labels, transform=transform)
        test_dataset = ImageClassificationDataset(test_image_paths, test_labels, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, valid_loader, test_loader, num_classes

    # Parse COCO Annotations
    def parseCocoAnnotations(self, annotation_file, base_path):
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
                labels.append(ann['category_id']) 

            return image_paths, labels, annotations['categories']  
        except Exception as e:
        # Get the traceback as a string
            traceback_str = traceback.format_exc()
            print(traceback_str)
            # Get the line number of the exception
            line_no = traceback.extract_tb(e.__traceback__)[-1][1]
            print(f"Exception occurred on line {line_no}")
            return str(e)


    # Create category mapping
    def createCategoryMapping(self, categories):
        try:
            category_mapping = {cat['id']: cat['name'] for cat in categories}
            return category_mapping, len(category_mapping)
        except Exception as e:
        # Get the traceback as a string
            traceback_str = traceback.format_exc()
            print(traceback_str)
            # Get the line number of the exception
            line_no = traceback.extract_tb(e.__traceback__)[-1][1]
            print(f"Exception occurred on line {line_no}")
            return str(e)


    # Create MobileNetV2 model
    def createModel(self):
        try:
            model = models.mobilenet_v2(weights='DEFAULT')  
            model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
            return model
        except Exception as e:
        # Get the traceback as a string
            traceback_str = traceback.format_exc()
            print(traceback_str)
            # Get the line number of the exception
            line_no = traceback.extract_tb(e.__traceback__)[-1][1]
            print(f"Exception occurred on line {line_no}")
            return str(e)


    # Train Model
    def trainModel(self):
        try:
            for epoch in range(self.num_epochs):
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                epoch_loss = running_loss / total
                epoch_acc = correct / total
                print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
                
                self.validateModel()
        except Exception as e:
        # Get the traceback as a string
            traceback_str = traceback.format_exc()
            print(traceback_str)
            # Get the line number of the exception
            line_no = traceback.extract_tb(e.__traceback__)[-1][1]
            print(f"Exception occurred on line {line_no}")
            return str(e)


    # Validate Model
    def validateModel(self):
        try:
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f'Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}')
        except Exception as e:
        # Get the traceback as a string
            traceback_str = traceback.format_exc()
            print(traceback_str)
            # Get the line number of the exception
            line_no = traceback.extract_tb(e.__traceback__)[-1][1]
            print(f"Exception occurred on line {line_no}")
            return str(e)


    # Test Model
    def testModel(self):
        try:
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            print(f'Test Accuracy: {accuracy:.4f}')
        except Exception as e:
        # Get the traceback as a string
            traceback_str = traceback.format_exc()
            print(traceback_str)
            # Get the line number of the exception
            line_no = traceback.extract_tb(e.__traceback__)[-1][1]
            print(f"Exception occurred on line {line_no}")
            return str(e)

    # Save Model
    def saveModel(self, path):
        try:
            torch.save(self.model.state_dict(), path)
        except Exception as e:
        # Get the traceback as a string
            traceback_str = traceback.format_exc()
            print(traceback_str)
            # Get the line number of the exception
            line_no = traceback.extract_tb(e.__traceback__)[-1][1]
            print(f"Exception occurred on line {line_no}")
            return str(e)


# Custom Dataset class
'''Class encapsulates the functionality required to load and preprocess images for a classification task in PyTorch.'''
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

