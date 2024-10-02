from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from requests import Session
from app.schema.response_schema import ResponseSchema
from app.modules.image_detection import image_detection_service
from config.database import getDb, msg


router = APIRouter(prefix="/api/image", tags = ["Object detection"])

# Detect object in an image 
@router.post("/object/detect", summary = "Detect objects in an image")
def imageDetect(background_tasks: BackgroundTasks, upload_file: UploadFile = File(...), db: Session = Depends(getDb)):
    detected_image = image_detection_service.imageDetect(background_tasks = background_tasks, upload_file = upload_file, db = db)
    
    if detected_image == 1:
            return ResponseSchema(status = False, response = msg['file_format_not_supported'], data = None)
    elif detected_image:
        return ResponseSchema(status = True, response = msg['object_detected'], data = detected_image)
    else:
        return ResponseSchema(status = False, response = msg['object_not_detected'], data = None)
    

