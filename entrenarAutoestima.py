import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display

def autoestima_entrenamiento(num_records=1000, test_size=0.2):
    """
    Genera datos sintéticos para evaluar un test de autoestima y entrena un modelo.

    Args:
        num_records (int): Cantidad de registros a generar.
        test_size (float): Porcentaje del conjunto de prueba (entre 0 y 1).

    Returns:
        tuple: (accuracy, conf_matrix) del modelo entrenado.
    """
    scores = np.random.randint(0, 31, num_records)

    labels = []
    for score in scores:
        if 0 <= score <= 10:
            labels.append('baja')
        elif 11 <= score <= 20:
            labels.append('media')
        elif 21 <= score <= 30:
            labels.append('alta')


    df_autoestima = pd.DataFrame({'Puntuación': scores, 'Etiqueta': labels})
    display(df_autoestima.head())

    label_distribution = df_autoestima['Etiqueta'].value_counts()
    display(label_distribution)


    X = df_autoestima[['Puntuación']]
    y = df_autoestima['Etiqueta']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, model

def guardar_modelo_autoestima(model, filename):
    """
    Guarda un modelo entrenado como un archivo pickle.

    Args:
        model: El modelo entrenado a guardar.
        filename (str): El nombre del archivo donde se guardará el modelo (con extensión .pkl).
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo guardado exitosamente como '{filename}'")
        return True
    except Exception as e:
        print(f"Error al guardar el modelo como '{filename}': {e}")
        return False
