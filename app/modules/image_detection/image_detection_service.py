import os
import traceback
import cv2
from ultralytics import YOLO
from requests import Session
import shutil
from fastapi import BackgroundTasks, HTTPException, UploadFile
from datetime import datetime
import pathlib
from app.models.image_detection_model import ImageModel
from dotenv import load_dotenv 
from sqlalchemy.orm import Session

load_dotenv()
BASE_URL = os.getenv("BASE_URL")

SUPPORTED_FORMATS = {'webp', 'png', 'dng', 'bmp', 'mpo', 'jpeg', 'tiff', 'tif', 'pfm', 'jpg'}

# build a new model from scratch
# model = YOLO("yolov8n.yaml") 

# load a pretrained model 
model = YOLO("yolov8n.pt") 

# train the model
# model.train(data="coco128.yaml", epochs = 3)  # train the model

# evaluate model performance on the validation set
# metrics = model.val() 

# Detect object in an image
def imageDetect(background_tasks: BackgroundTasks, upload_file: UploadFile, db: Session):
    try:
        file_extension = os.path.splitext(upload_file.filename)[1][1:].lower()  
        if file_extension not in SUPPORTED_FORMATS:
            return 1  
        
        # Directory for saving uploaded images
        upload_dir="uploads"
      
        # Directory for saving predicted images
        save_dir = "predicted_images"
        
        if not upload_file.filename:
            raise HTTPException(status_code = 400, detail = "No selected file")
        
        filename = upload_file.filename
        timestamp = int(datetime.now().timestamp())
        file_extension = pathlib.Path(filename).suffix
        unique_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)

        with open(file_path, "wb") as buffer:
            buffer.write(upload_file.file.read())

        print(f"Uploaded file saved at: {file_path}")

        # Background processing function
        def processImage():
            results = model(file_path, save=True)
            predicted_image_path = None

            for result in results:
                default_save_dir = result.save_dir  
                print(f"Default save directory: {default_save_dir}")

                for file_name in os.listdir(default_save_dir):
                    full_file_name = os.path.join(default_save_dir, file_name)
                    predicted_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{file_extension}"
                    destination_file = os.path.join(save_dir, predicted_filename)

                    print(f"Moving file from {full_file_name} to {destination_file}")
                    if os.path.isfile(full_file_name):
                        if os.path.exists(destination_file):
                            print(f"File {file_name} already exists in {save_dir}")
                        else:
                            shutil.move(full_file_name, destination_file)
                            print(f"Moved {file_name} to {save_dir}")
            
            shutil.rmtree('runs')
            predicted_image_path = os.path.join(save_dir, predicted_filename)
            print(f"Predicted image path: {predicted_image_path}")

            if not os.path.exists(predicted_image_path):
                print(f"Error: Predicted image file does not exist at {predicted_image_path}")

            new_image = ImageModel(original_image_path = file_path, predicted_image_path = predicted_image_path)
            db.add(new_image)
            db.commit()
            db.refresh(new_image)

            # Display the predicted image
            if os.path.exists(predicted_image_path):
                print(f"Image file exists at {predicted_image_path}")
                image = cv2.imread(predicted_image_path)
                
                if image is not None:
                    cv2.imshow('Displayed Image', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Error loading the predicted image")
            else:
                print("Predicted image file does not exist")

            return predicted_image_path
        
        # Add task to background queue
        background_tasks.add_task(processImage)

        predicted_image_url = f"{BASE_URL}{save_dir}/{unique_filename}" 
        response = {
            "image_profile_url": predicted_image_url
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


