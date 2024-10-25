import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from config.database import engine, Base
from config.database import SessionLocal
from app.modules.compare_image import compare_image_route
from app.modules.compare_image_model import compare_image_model_route
from app.modules.image_compression import image_compression_route
from app.modules.image_detection import image_detection_route
from app.modules.image_classification import image_classification_route
from app.modules.context_analyze import context_analyze_route
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Ensure the directory exists, if not create it
upload_images_dir = os.path.join(os.getcwd(), "uploads")
compressed_images_dir = os.path.join(os.getcwd(), "compressed_images")
predicted_images_dir = os.path.join(os.getcwd(), "predicted_images")

os.makedirs(upload_images_dir, exist_ok = True)
os.makedirs(compressed_images_dir, exist_ok = True)
os.makedirs(predicted_images_dir, exist_ok = True)


app.mount("/predicted_images", StaticFiles(directory = "predicted_images"), name = "predicted_images")
app.mount("/compressed_images", StaticFiles(directory = "compressed_images"), name = "compressed_images")


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

Base.metadata.create_all(bind = engine)
session = SessionLocal()

app.include_router(compare_image_route.router)
app.include_router(compare_image_model_route.router)
app.include_router(image_compression_route.router)
app.include_router(image_detection_route.router)
app.include_router(image_classification_route.router)
app.include_router(context_analyze_route.router)


if __name__ == '__main__':
    uvicorn.run("main:app", host = '192.168.1.53', port = 8000, log_level = "info", reload = True)
    print("running")
