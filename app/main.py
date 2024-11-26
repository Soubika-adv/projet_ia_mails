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
@app.put('/cluster/{id_payeur}')
def predict_and_update(db : Session = Depends(get_db)):
    email_entries = db.query(EmailPrediction).all()

    #valid_email_entries = [entry for entry in email_entries if is_valid(entry.mail_payeur)]
    all_emails = [entry.mail_payeur for entry in email_entries if is_valid(entry.mail_payeur)]
    
    if not all_emails:
        return {"message": "Aucun e-mail valide trouvé."}


    all_embeddings = []

    # Traitement par lot pour éviter des erreurs de mémoire
    for i in range(0, len(all_emails), batch_size):
        batch_emails = all_emails[i:i + batch_size]
        
        # Tokenizer et récupération des embeddings avec Camembert
        inputs = tokenizer(batch_emails, padding=True, truncation=True, max_length=64, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(batch_embeddings)

    # Étape 3: Application du clustering K-Means avec 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_predictions = kmeans.fit_predict(all_embeddings)

    # Étape 4: Mise à jour des entrées dans la base de données
    valid_idx = 0
    for email_entry in email_entries:
        if is_valid(email_entry.mail_payeur):
            email_entry.cluster_prediction = int(cluster_predictions[valid_idx])
            valid_idx += 1
        else:
            email_entry.cluster_prediction = None

        db.commit()

    return {"message": f"{valid_idx} adresses ont été mises à jour avec des labels de cluster."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8002)


