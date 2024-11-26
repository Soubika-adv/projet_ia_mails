from sqlalchemy.orm import Session
import uvicorn
import torch
from fastapi import FastAPI, Depends
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import models
from models import Base, EmailPrediction
from database import engine, SessionLocal
import re
from sklearn.cluster import KMeans

models.Base.metadata.create_all(bind = engine) # initialise les tables de la base de données


best_model = "C:/Users/Advensys/Desktop/dev/projet_mail/inference_camembert/app/best_2_model"
model = CamembertForSequenceClassification.from_pretrained(best_model)
tokenizer = CamembertTokenizer.from_pretrained(best_model)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def is_valid(email):
    if not email:
        return False
    email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    return bool(re.match(email_regex, email))

batch_size = 128
@app.put('/prediction/{id_payeur}')
def predict_and_update(db : Session = Depends(get_db)):
    email_entries = db.query(EmailPrediction).all()

    for i in range(0, len(email_entries), batch_size):
        batch_entries = email_entries[i:i+batch_size]
        adresses_mail = [entry.mail_payeur for entry in batch_entries]

        invalid_emails = []
        for idx, email in enumerate(adresses_mail):
            if not is_valid(email):
                batch_entries[idx].mail_payeur = None
                invalid_emails.append(idx)
        
        valid_emails = [adresses_mail for idx, email in enumerate(adresses_mail) if idx not in invalid_emails]
        if valid_emails:
            inputs = tokenizer(adresses_mail, padding = True, truncation = True, max_length = 64, return_tensors = 'pt')

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, axis = 1).cpu().numpy()

            valid_idx = 0
            for idx, email_entry in enumerate(batch_entries):
                if idx not in invalid_emails:
                    email_entry.prediction = predictions[valid_idx]
                    db.commit()

    return {"message": f"{len(email_entries)} adresses ont été prédites et mises à jour."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8002)


