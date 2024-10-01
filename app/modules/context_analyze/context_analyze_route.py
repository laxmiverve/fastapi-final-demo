from fastapi import APIRouter, File, UploadFile
from app.schema.response_schema import ResponseSchema
from config.database import msg
from app.modules.context_analyze import context_analyze_service


router = APIRouter(prefix="/api/image", tags = ["Text extract"])

# Extracting text from images
@router.post("/text/extract", summary = "Extracting text from an image")
def extractText(upload_file: UploadFile = File(...)):
        extraction_result = context_analyze_service.extractText(upload_file = upload_file)
        
        if extraction_result == 1:
            return ResponseSchema(status = False, response = msg['file_format_not_supported'], data = None)
        elif extraction_result:
            return ResponseSchema(status = True, response = msg['contextualization_successful'], data = extraction_result)
        else:
            return ResponseSchema(status = False, response = msg['contextualization_failed'], data = None)
