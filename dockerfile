# Utilise une image Python de base
FROM python:3.9-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers de ton projet dans le conteneur
COPY . /app

# Installe les dépendances du projet
RUN pip install --no-cache-dir -r requirements.txt

# Installe YOLOv5 et ses dépendances spécifiques
RUN pip install --no-cache-dir torch torchvision torchaudio \
    && git clone https://github.com/ultralytics/yolov5.git \
    && cd yolov5 && pip install -r requirements.txt

# Expose le port pour Streamlit (par défaut 8501)
EXPOSE 8501

# Commande pour lancer l'application Streamlit
CMD ["streamlit", "run", "app.py"]
