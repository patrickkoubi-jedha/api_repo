FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Patrick"
LABEL description="FastAPI Iris Prediction API"

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Création d'un utilisateur non-root
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Copie des requirements en premier (pour optimiser le cache Docker)
COPY --chown=user requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY --chown=user . .

# Exposition du port
EXPOSE 8000

# Commande de démarrage (recommandée pour FastAPI)
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
