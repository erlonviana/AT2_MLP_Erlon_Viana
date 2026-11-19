from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.schemas import PredicaoInput, PredicaoOutput


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o pipeline UMA vez no startup
    app.state.modelo = joblib.load("models/pipeline_rede_neural.joblib")
    yield
    app.state.modelo = None


app = FastAPI(
    title="API MLP — Bank Marketing",
    description="Predição de adesão a depósito a prazo com pipeline MLP",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/diagnostico/modelo")
def diagnostico():
    return {
        "status": "operacional_v2",
        "framework": "FastAPI-Production",
        "modelo": "MLPClassifier"
    }


@app.post("/predict", response_model=PredicaoOutput)
def predict(payload: PredicaoInput):
    modelo = app.state.modelo
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    # DataFrame com as mesmas colunas do treino
    df = pd.DataFrame([payload.model_dump()])

    pred = modelo.predict(df)[0]
    proba = modelo.predict_proba(df)[0]
    score = float(proba.max())

    return PredicaoOutput(
        categoria_prevista=str(pred),
        score_confianca=score
    )