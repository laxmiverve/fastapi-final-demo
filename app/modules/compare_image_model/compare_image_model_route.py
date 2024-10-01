from fastapi import APIRouter, File, UploadFile
from app.schema.response_schema import ResponseSchema
from config.database import msg
from app.modules.compare_image_model import compare_image_model_service


router = APIRouter(prefix="/api/image", tags = ["Image compare using ML model"])

# Compare images similar or not using model
@router.post("/compare/model", summary = "Compare images similar or not using model")
async def compareImagesModel(upload_file1: UploadFile = File(...), upload_file2: UploadFile = File(...)):
    compare_result = await compare_image_model_service.compareImagesModel(upload_file1 = upload_file1, upload_file2 = upload_file2)
    
    if compare_result == 1:
        return ResponseSchema(status = False, response = msg['file_format_not_supported'], data = None)
    elif compare_result['similarity_score'] > 90:
        return ResponseSchema(status = True, response = msg['similar_image'], data = compare_result)
    else:
        return ResponseSchema(status = False, response = msg['not_similar_image'], data = compare_result)
