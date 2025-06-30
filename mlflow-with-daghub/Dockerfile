# Étape 1: Utiliser une image Python officielle comme base
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances
# On ajoute joblib qui est maintenant nécessaire
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir joblib

# Copier tout le reste du code de l'application dans le conteneur
COPY . .

# Exposer le port sur lequel l'API s'exécutera
EXPOSE 8000

# La commande pour démarrer l'API quand le conteneur se lance
# Elle exécute le serveur Uvicorn qui fait tourner le fichier api.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]