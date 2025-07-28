from fastapi import FastAPI
import pickle
import pandas as pd

from personalidadModel import PerfilPersonalidad

# Inicializar la aplicación FastAPI
app = FastAPI()


# Cargar el modelo entrenado
def cargar_modelo(ruta: str):
    with open(ruta, 'rb') as file:
        return pickle.load(file)


# Endpoint informativo
@app.get("/info")
def obtener_info():
    return {"mensaje": "API para predicción de test"}

# Endpoint de predicción
@app.post("/predecir_personalidad_rasgos/")
def predecir_personalidad(perfil: PerfilPersonalidad):
    """
    Realiza una predicción de perfil de personalidad.
    """
    modelo = cargar_modelo('modelo_personalidad.pkl')

    datos_df = pd.DataFrame([perfil.dict()])
    prediccion = modelo.predict(datos_df)
    return {"perfil": prediccion[0]}
