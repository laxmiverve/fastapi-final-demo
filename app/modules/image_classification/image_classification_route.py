from fastapi import APIRouter, File, UploadFile
from app.schema.response_schema import ResponseSchema
from config.database import msg
from app.modules.image_classification import image_classification_service


router = APIRouter(prefix="/api/image", tags = ["Image classificaion"])

@router.post("/classify", summary = "Classify the similar image")
def predictImage(upload_file: UploadFile = File(...)):
    image_classify = image_classification_service.predictImage(upload_file = upload_file)
    
    if image_classify == 1:
            return ResponseSchema(status = False, response = msg['file_format_not_supported'], data = None)
    elif image_classify:
        return ResponseSchema(status = True, response = msg['image_classification_successful'], data = image_classify)
    else:
        return ResponseSchema(status = False, response = msg["image_classification_error"], data = None)
