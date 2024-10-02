import traceback
import cv2
from fastapi import UploadFile
import pytesseract
import numpy as np
import os 

SUPPORTED_FORMATS = {'webp', 'png', 'dng', 'bmp', 'mpo', 'jpeg', 'tiff', 'tif', 'pfm', 'jpg'}


# Extracting text from an image 
def extractText(upload_file: UploadFile):
    try:
        file_extension = os.path.splitext(upload_file.filename)[1][1:].lower() 
        if file_extension not in SUPPORTED_FORMATS:
            return 1  
        
        # Read the image file into memory
        file_bytes = np.frombuffer(upload_file.file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("Image not found or could not be loaded")
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(gray_image)
        
        response = {
            "extracted_text": text
        }
        return response
    except Exception as e:
        # Get the traceback as a string
        traceback_str = traceback.format_exc()
        print(traceback_str)
        # Get the line number of the exception
        line_no = traceback.extract_tb(e.__traceback__)[-1][1]
        print(f"Exception occurred on line {line_no}")
        return str(e)


