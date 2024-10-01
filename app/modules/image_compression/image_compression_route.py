from fastapi import APIRouter, File, UploadFile
from app.schema.response_schema import ResponseSchema
from config.database import msg
from app.modules.image_compression import image_compression_service

router = APIRouter(prefix="/api/image", tags = ["Image compresser"])

# Compress the image
@router.post("/compress", summary = "Compress an image")
def compressImage(upload_file: UploadFile = File(...)):
        compressed_image = image_compression_service.compressImage(upload_file = upload_file)
        
        if compressed_image == 1:
            return ResponseSchema(status = False, response = msg['file_format_not_supported'], data = None)
        if compressed_image:
            return ResponseSchema(status = True, response = msg['compression_success'], data = compressed_image)
        else:
            return ResponseSchema(status = True, response = msg['compression_failed'], data = None)
        
