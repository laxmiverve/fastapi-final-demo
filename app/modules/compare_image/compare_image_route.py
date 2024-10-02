from fastapi import APIRouter, File, UploadFile
from app.schema.response_schema import ResponseSchema
from config.database import msg
from app.modules.compare_image import compare_image_service


router = APIRouter(prefix="/api/image", tags = ["Image compare using SSIM"])

# Compare images similar or not using SSIM (Structural similarity index)
@router.post("/compare/ssim", summary = "Compare images similar or not using SSIM")
async def compareImages(upload_file1: UploadFile = File(...), upload_file2: UploadFile = File(...)):
    compare_result = await compare_image_service.compareImages(upload_file1 = upload_file1, upload_file2 = upload_file2)
    data = {"similarity_percentage": compare_result}
    similarity = data["similarity_percentage"]  

    if compare_result == 1:
        return ResponseSchema(status = False, response = msg['file_format_not_supported'], data = None)
    elif similarity > 90:
        return ResponseSchema(status = True, response = msg['high_confidence'], data = data)
    elif similarity > 80 and similarity <= 90:
        return ResponseSchema(status = False, response = msg['almost_similar'], data = data)
    elif similarity > 70 and similarity <= 80:
        return ResponseSchema(status = False, response = msg['unsure_similarity'], data = data)
    else:
        return ResponseSchema(status = False, response = msg['not_similar_image'], data = data) 