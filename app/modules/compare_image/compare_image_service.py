from fastapi import UploadFile
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import os

# Supported image formats
SUPPORTED_FORMATS = {'webp', 'png', 'dng', 'bmp', 'mpo', 'jpeg', 'tiff', 'tif', 'pfm', 'jpg'}

# Convert the image bytes to a grayscale image using OpenCV
def preprocessImage(image_bytes):
    try:
        # Convert bytes to PIL image and then to grayscale using OpenCV
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image
    except Exception as e:
        print("An exception occurred during image preprocessing:", str(e))
        return None  # Return None in case of error

# Compare images using SSIM (Structural Similarity Index)
async def compareImages(upload_file1: UploadFile, upload_file2: UploadFile):
    try:
        uploaded_files = [upload_file1, upload_file2]
        processed_images = []

        for upload_file in uploaded_files:
            # Get the file extension and check if it's supported
            file_extension = os.path.splitext(upload_file.filename)[1][1:].lower()  # Get the file extension without the dot
            if file_extension not in SUPPORTED_FORMATS:
                return 1  # Return 1 for unsupported file type
            
            image_bytes = await upload_file.read()
            gray_image = preprocessImage(image_bytes)
            if gray_image is None:
                return 1   
            
            processed_images.append(gray_image)

        # Extract the preprocessed images
        gray_image1, gray_image2 = processed_images
        
        # Resize the second image to match the first image's dimensions for comparison
        gray_image2_resized = cv2.resize(gray_image2, (gray_image1.shape[1], gray_image1.shape[0]))

        # Compute SSIM between the two images
        similarity, diff = ssim(gray_image1, gray_image2_resized, full=True)

        similarity_percentage = similarity * 100

        if similarity_percentage > 90:
            message = "The images are considered similar with high confidence."
        elif similarity_percentage > 80 and similarity_percentage <= 90:
            message = "The images are almost similar but not completely the same."
        elif similarity_percentage > 70 and similarity_percentage <= 80:
            message = "Not sure if the images are the same or not."
        else:
            message = "The images are not considered similar."
        
        return {
            "similarity_score": round(float(similarity_percentage), 2),
            "message": message
        }
    except Exception as e:
        print("An exception occurred:", str(e))


'''
SSIM calculated using the following components:
- Luminance: Measures the brightness of the images.
- Contrast: Measures the difference in color or intensity.
- Structure: Measures the patterns of pixel intensities.
'''

