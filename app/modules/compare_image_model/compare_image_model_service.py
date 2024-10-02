import traceback
from fastapi import UploadFile
import torch
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PerceiverImageProcessor, PerceiverForImageClassificationLearned
import os


# Load the Perceiver model and feature extractor
feature_extractor = PerceiverImageProcessor.from_pretrained("addy88/perceiver_image_classifier")
model = PerceiverForImageClassificationLearned.from_pretrained("addy88/perceiver_image_classifier")


SUPPORTED_FORMATS = {'webp', 'png', 'dng', 'bmp', 'mpo', 'jpeg', 'tiff', 'tif', 'pfm', 'jpg'}

# Process the image and extract the feature from image 
def preprocessAndExtractFeatures(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = feature_extractor(images = image, return_tensors = "pt")
        with torch.no_grad():
            outputs = model(**inputs)
        features = outputs.logits.numpy()
        return features
    except Exception as e:
        # Get the traceback as a string
        traceback_str = traceback.format_exc()
        print(traceback_str)
        # Get the line number of the exception
        line_no = traceback.extract_tb(e.__traceback__)[-1][1]
        print(f"Exception occurred on line {line_no}")
        return str(e)



# Compare images similar or not using model
async def compareImagesModel(upload_file1: UploadFile, upload_file2: UploadFile):
    try:
        for upload_file in [upload_file1, upload_file2]:
            file_extension = os.path.splitext(upload_file.filename)[1][1:].lower() 
            if file_extension not in SUPPORTED_FORMATS:
                return 1  
            
        image_bytes1 = await upload_file1.read()
        image_bytes2 = await upload_file2.read()

        features1 = preprocessAndExtractFeatures(image_bytes1)
        features2 = preprocessAndExtractFeatures(image_bytes2)
        
        if features1 is None or features2 is None:
            return 1  
        
        # Compute cosine similarity between features
        similarity = cosine_similarity(features1, features2)[0][0]
        similarity_percentage = similarity * 100
        similarity_percentage = round(float(similarity_percentage), 2)

        return similarity_percentage
    except Exception as e:
        # Get the traceback as a string
        traceback_str = traceback.format_exc()
        print(traceback_str)
        # Get the line number of the exception
        line_no = traceback.extract_tb(e.__traceback__)[-1][1]
        print(f"Exception occurred on line {line_no}")
        return str(e)

