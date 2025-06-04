from sqlalchemy import Column, String
from models.base import Base

class User(Base):
    __tablename__ = 'users'
    userid = Column(String(20), primary_key=True, index=True)
    userpw = Column(String(64), nullable=False)
    name = Column(String(10), nullable=False)
    phone_number = Column(String(20), nullable=False)
