import traceback
from fastapi import UploadFile
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import os

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
        # Get the traceback as a string
        traceback_str = traceback.format_exc()
        print(traceback_str)
        # Get the line number of the exception
        line_no = traceback.extract_tb(e.__traceback__)[-1][1]
        print(f"Exception occurred on line {line_no}")
        return str(e)

# Compare images using SSIM (Structural Similarity Index)
async def compareImages(upload_file1: UploadFile, upload_file2: UploadFile):
    try:
        uploaded_files = [upload_file1, upload_file2]
        processed_images = []

        for upload_file in uploaded_files:
            # Get the file extension and check if it's supported
            file_extension = os.path.splitext(upload_file.filename)[1][1:].lower() 
            if file_extension not in SUPPORTED_FORMATS:
                return 1 
            
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
        similarity_percentage = round(float(similarity * 100), 2)
        return similarity_percentage
    except Exception as e:
        # Get the traceback as a string
        traceback_str = traceback.format_exc()
        print(traceback_str)
        # Get the line number of the exception
        line_no = traceback.extract_tb(e.__traceback__)[-1][1]
        print(f"Exception occurred on line {line_no}")
        return str(e)



'''
SSIM calculated using the following components:
- Luminance: Measures the brightness of the images.
- Contrast: Measures the difference in color or intensity.
- Structure: Measures the patterns of pixel intensities.
'''

