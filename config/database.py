import os
import json
import traceback
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

metadata = MetaData()

Base = declarative_base()

load_dotenv()

user = os.getenv("DB_USER")
database = os.getenv("DB_NAME")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")

MYSQL_URL = f"mysql+pymysql://{user}:{password}@{host}/{database}"
engine = create_engine(MYSQL_URL)
SessionLocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)

file = open(os.getcwd() + '/response_message.json')
msg = json.load(file)

def getDb():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        # Get the traceback as a string
        traceback_str = traceback.format_exc()
        print(traceback_str)
        # Get the line number of the exception
        line_no = traceback.extract_tb(e.__traceback__)[-1][1]
        print(f"Exception occurred on line {line_no}")
        return str(e), e
    finally:
        db.close()