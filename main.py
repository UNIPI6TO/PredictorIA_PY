from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Crear una instancia de la aplicación FastAPI
app = FastAPI()

# Definir el modelo de datos para la entrada de la API
class perfilPersonalidad(BaseModel):
    Extroversión: float
    Responsabilidad: float
    Amabilidad: float
    Neuroticismo: float
    Apertura: float

# Cargar el modelo entrenado
with open('modelo_personalidad.pkl', 'rb') as file:
    modelo = pickle.load(file)

@app.get("/info")
def read_root():
    return {"mensaje": "API para predicción de personalidad"}

@app.post("/predecir_personalidad_rasgos/")
def predecir_personalidad(perfil: perfilPersonalidad):
    """
    Realiza una predicción de perfil de personalidad.
    """
    # Convertir los datos de entrada a un DataFrame de pandas
    datos_df = pd.DataFrame({perfil.dict()})

    # Realizar la predicción
    prediccion = modelo.predict(datos_df)

    # Devolver la predicción
    return {"perfil": prediccion[0]}