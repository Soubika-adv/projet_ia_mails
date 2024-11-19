from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class EmailPrediction(Base):
    __tablename__ = "gcf_payeur"

    id_payeur = Column(Integer, primary_key=True, index = True)
    mail_payeur = Column(String, index = True)
    prediction = Column(Integer)