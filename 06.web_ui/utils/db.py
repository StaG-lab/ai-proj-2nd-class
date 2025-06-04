import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

CONFIG_PATH = "./db_config.json"

# JSON 파일 로드
with open(CONFIG_PATH) as f:
    config = json.load(f)

DB_USER = config['DB_USER']
DB_PASSWORD = config['DB_PASSWORD']
DB_HOST = config['DB_HOST']
DB_PORT = config['DB_PORT']
DB_NAME = config['DB_NAME']

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
