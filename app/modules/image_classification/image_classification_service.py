import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from fastapi import UploadFile
import os

SUPPORTED_FORMATS = {'webp', 'png', 'dng', 'bmp', 'mpo', 'jpeg', 'tiff', 'tif', 'pfm', 'jpg'}

# model_path = 'mobilenet_v2_image_classifier.pth'
model_path = 'mobilenet_v3_image_classifier.pth'

def predictImage(upload_file: UploadFile):
    try:
        file_extension = os.path.splitext(upload_file.filename)[1][1:].lower()  # Get the file extension without the dot
        if file_extension not in SUPPORTED_FORMATS:
            return 1  # Return 1 for unsupported file type
        
        
        # Load the COCO annotations to extract categories
        # dataset_path = 'datasets/image-classify.v1i.coco/train/_annotations.coco.json'
        dataset_path = 'datasets/classify-image.v1i.coco/train/_annotations.coco.json'
        
        with open(dataset_path, 'r') as f:
            annotations = json.load(f)
            
        # In context of COCO annotations, cat typically refers to each category in the dataset
        category_mapping = {cat['id']: cat['name'] for cat in annotations['categories']}
        
        # Load the trained model
        num_classes = len(category_mapping)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = models.mobilenet_v2(weights = 'IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)

        # Preprocess the uploaded image
        IMG_SIZE = (224, 224)
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
        
        image = Image.open(upload_file.file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Make a prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim = 1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_label = predicted.item()
        confidence_score = confidence.item() * 100
        confidence_score = round(confidence_score, 2)

        if predicted_label in category_mapping:
            predicted_class = category_mapping[predicted_label] # Map category id to category name
        else:
            predicted_class = "Category not found"
        
        response = {
            "The given image was": predicted_class
        }
        
        return response
    except Exception as e:
        print("An exception occurred:", str(e))
