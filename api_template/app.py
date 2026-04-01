import uvicorn
from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ====================== CONFIGURATION MLFLOW ======================
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
print("[INFO] MLflow tracking URI configuré :", os.environ["MLFLOW_TRACKING_URI"])

# Chargement du modèle
registered_model = "iris_classifier"
model_alias = "challenger"
model_uri = f"models:/{registered_model}@{model_alias}"

try:
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modèle '{registered_model}@{model_alias}' chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    loaded_model = None

# ====================== FASTAPI APP ======================
app = FastAPI(title="Iris Prediction API", version="1.0")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probability: float

@app.get("/")
async def root():
    return {"message": "Iris Prediction API is running. Go to /docs"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Préparer les données
    input_data = pd.DataFrame([{
        "sepal length (cm)": features.sepal_length,
        "sepal width (cm)": features.sepal_width,
        "petal length (cm)": features.petal_length,
        "petal width (cm)": features.petal_width
    }])

    # Prédiction
    prediction = loaded_model.predict(input_data)[0]
    probability = float(loaded_model.predict_proba(input_data).max())

    class_names = ["setosa", "versicolor", "virginica"]

    return {
        "prediction": int(prediction),
        "class_name": class_names[prediction],
        "probability": probability
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
