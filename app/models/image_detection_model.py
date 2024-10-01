from sqlalchemy import Column, Integer, String
from config.database import Base

class ImageModel(Base):
    __tablename__ = "image_detection"

    id = Column(Integer, primary_key = True, index = True)
    original_image_path = Column(String, nullable = False)
    predicted_image_path = Column(String, nullable = False)