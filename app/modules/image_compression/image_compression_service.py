from datetime import datetime
import pathlib
import traceback
from dotenv import load_dotenv
from fastapi import UploadFile
from PIL import Image
import os
import io

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
SUPPORTED_FORMATS = {'webp', 'png', 'dng', 'bmp', 'mpo', 'jpeg', 'tiff', 'tif', 'pfm', 'jpg'}


# Compress the image
def compressImage(upload_file: UploadFile):
    try:
        file_extension = os.path.splitext(upload_file.filename)[1][1:].lower() 
        if file_extension not in SUPPORTED_FORMATS:
            return 1  
        
        base_dir = os.getcwd()
        quality = 100
        resize_factor = 0.5
        step = 5 

        image_bytes = upload_file.file.read()
        filename = upload_file.filename

        # Create an in-memory image object
        image = Image.open(io.BytesIO(image_bytes))

        original_image_size_kb = len(image_bytes) / 1024

        timestamp = int(datetime.now().timestamp())
        file_extension = pathlib.Path(filename).suffix
        unique_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{file_extension}"
        relative_dir = 'compressed_images'
        output_dir = os.path.join(base_dir, relative_dir)

        output_path = os.path.join(output_dir, unique_filename)

        # Resize the image
        if resize_factor != 1.0:
            width, height = image.size
            new_size = (int(width * resize_factor), int(height * resize_factor))
            image = image.resize(new_size, Image.LANCZOS)


        while quality > 10:
            # Save image with current quality and check the size
            image.save(output_path, format = image.format, quality = quality, optimize = True, lossless = False)
            size_kb = os.path.getsize(output_path) / 1024

            if size_kb <= 100:  
                break
            quality = quality - step

        # extract relative path
        relative_path = os.path.relpath(output_path, start = base_dir)

        image_url = f"{BASE_URL}{relative_path}"

        response =  {
            "original_image_size_kb": f"{original_image_size_kb:.2f} KB",
            "compressed_image_size_kb": f"{size_kb:.2f} KB",
            "image_url": image_url
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

