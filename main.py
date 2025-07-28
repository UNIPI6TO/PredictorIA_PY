from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inicializar la aplicación FastAPI
app = FastAPI()

# Modelo de entrada para la API
class PerfilPersonalidad(BaseModel):
    Extroversión: float
    Responsabilidad: float
    Amabilidad: float
    Neuroticismo: float
    Apertura: float

# Cargar el modelo entrenado
def cargar_modelo(ruta: str):
    with open(ruta, 'rb') as file:
        return pickle.load(file)

modelo = cargar_modelo('modelo_personalidad.pkl')

# Endpoint informativo
@app.get("/info")
def obtener_info():
    return {"mensaje": "API para predicción de personalidad"}

# Endpoint de predicción
@app.post("/predecir_personalidad_rasgos/")
def predecir_personalidad(perfil: PerfilPersonalidad):
    """
    Realiza una predicción de perfil de personalidad.
    """
    datos_df = pd.DataFrame([perfil.dict()])
    prediccion = modelo.predict(datos_df)
    return {"perfil": prediccion[0]}
