from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

username = 'sqladv'
password = 'sqladv'
hostname = '127.0.0.1'
port = '3306'  
database = 'umihf_gcf'

# Crée une chaîne de connexion pour accéder à une base de données MySQL
DATABASE_URL = f'mysql+pymysql://{username}:{password}@{hostname}:{port}/{database}'

# Initialise une connexion avec SQLAlchemy pour interagir avec la base de données
engine = create_engine(DATABASE_URL, echo = True)


# Configure une 'usine' de sessions de connexion
SessionLocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)


# Sert de classe de base pour toutes les classes de modèles SQLAlchemy
Base = declarative_base()
