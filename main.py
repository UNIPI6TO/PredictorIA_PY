
import os
import shutil
import pickle
import pandas as pd

from entrenarModeloPersonalidad import PerfiladorPersonalidad
from personalidadModel import PerfilPersonalidad
from fastapi import FastAPI, HTTPException

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

    items = prediccion[0].split(', ')

    data_dict = {}
    for item in items:
        # Separamos el valor y la característica
        parts = item.split(' en ')
        if len(parts) == 2:
            value = parts[0].strip()
            characteristic = parts[1].strip()
            data_dict[characteristic] = value

    return  data_dict

@app.get("/entrenar_personalidad_rasgos/")
def entrenar_personalidad(muestras: int = 5000, porcentajePruebas: float = 0.2):
    # Crea una instancia de la clase
    perfilador = PerfiladorPersonalidad(random_state=77)
    # 1. Generar datos sintéticos
    print("Generando 5010 muestras por defecto de datos sintéticos...")

    datos_personalidad = perfilador.generar_datos_sinteticos(n_samples=muestras)

    # 2. Entrenar el modelo
    print("\nEntrenando el modelo de clasificación...")
    perfilador.entrenar_modelo(datos_personalidad, test_size=porcentajePruebas)

    precision = perfilador.evaluar_modelo()
    print(f"\nPrecisión del modelo en el conjunto de prueba: {precision:.4f}")

    # 5. Guardar el modelo entrenado
    nombre_archivo_modelo = "modelo_personalidad.pkl"
    temp_dir_in_root = os.path.join(os.getcwd(), "temp")

    try:
        os.makedirs(temp_dir_in_root, exist_ok=True)
        print(f"Directorio creado o ya existe: {temp_dir_in_root}")
    except OSError as e:
        print(f"Error al crear el directorio {temp_dir_in_root}: {e}")
        # Puedes decidir si quieres continuar o salir si el directorio no se puede crear
        return {"error": f"No se pudo crear el directorio: {e}"}

    ruta_guardado = os.path.join(temp_dir_in_root, nombre_archivo_modelo)  # Esto es más robusto para obtener la ruta actual

    with open(ruta_guardado, 'wb') as f:
        pickle.dump(perfilador.modelo, f)
    print(f"Modelo guardado exitosamente en {ruta_guardado}")
    return {"precision": precision, "modelo_path": "temp\\"+nombre_archivo_modelo}

@app.get("/implementar_modelo/")
def implementar_modelo(nombre_archivo: str) :
    """
    Mueve el archivo 'modelo_personalidad.pkl' de la carpeta 'temp' a la raíz del proyecto.
    Si el archivo ya existe en la raíz, lo reemplaza.
    """

    ruta_origen = os.path.join("temp", nombre_archivo)
    ruta_destino = nombre_archivo  # La raíz del proyecto

    if not os.path.exists(ruta_origen):
        print(f"Error: El archivo de origen '{ruta_origen}' no se encontró.")
        raise HTTPException(status_code=404, detail="Falta entrenar al modelo.")

    try:
        # shutil.move handles moving and overwriting the destination if it exists.
        shutil.move(ruta_origen, ruta_destino)
        print(f"'{nombre_archivo}' movido exitosamente de 'temp/' a la raíz del proyecto.")
        return {"estadoImplementacion": "Implementado"}
    except Exception as e:
        print(f"Error al mover el archivo '{nombre_archivo}': {e}")
        raise HTTPException(status_code=500, detail="Modelo no implementado "+ e.message +".")

