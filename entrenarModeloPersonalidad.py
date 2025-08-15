import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class PerfiladorPersonalidad:
    random_state: int

    """
    Clase para generar datos sintéticos de perfiles de personalidad,
    entrenar un modelo de clasificación y evaluar su rendimiento.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializa el PerfiladorPersonalidad.

        Args:
            random_state (int): Semilla para la reproducibilidad de los resultados.
        """
        self.random_state = random_state
        self.modelo = None  # El modelo se inicializará después del entrenamiento
        self.rasgos = ["Extroversión", "Responsabilidad", "Amabilidad", "Neuroticismo", "Apertura"]

    def generar_etiqueta(self, fila: pd.Series) -> str:
        """
        Método auxiliar (privado) para generar etiquetas de perfil de personalidad
        basadas en los valores de los rasgos.
        """
        etiquetas = []
        for i, valor in enumerate(fila):
            if valor >= 4:
                etiquetas.append(f"Alto en {self.rasgos[i]}")
            elif valor <= 2:
                etiquetas.append(f"Bajo en {self.rasgos[i]}")
            else:
                etiquetas.append(f"Medio en {self.rasgos[i]}")
        return ", ".join(etiquetas)

    def generar_datos_sinteticos(self, n_samples: int ) -> pd.DataFrame:
        """
        Genera un DataFrame con datos sintéticos de perfiles de personalidad.

        Args:
            n_samples (int): Número de muestras de datos sintéticos a generar.

        Returns:
            pd.DataFrame: DataFrame con los rasgos y los perfiles generados.
        """
        np.random.seed(self.random_state)
        X = np.random.randint(1, 6, size=(n_samples, 5))  # 5 rasgos Big Five

        df = pd.DataFrame(X, columns=self.rasgos)
        df["perfil"] = df.apply(self.generar_etiqueta, axis=1)
        return df

    def entrenar_modelo(self, df: pd.DataFrame, test_size: float ):
        """
        Entrena un modelo RandomForestClassifier con los datos proporcionados.

        Args:
            df (pd.DataFrame): DataFrame que contiene los rasgos y la columna 'perfil'.
            test_size (float): Proporción del conjunto de datos a usar para la prueba.
        """
        if "perfil" not in df.columns:
            raise ValueError("El DataFrame debe contener una columna 'perfil'.")

        X = df.drop("perfil", axis=1)
        y = df["perfil"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.modelo = RandomForestClassifier(random_state=self.random_state)
        self.modelo.fit(self.X_train, self.y_train)
        print("Modelo entrenado exitosamente.")

    def evaluar_modelo(self) -> float:
        """
        Evalúa la precisión del modelo entrenado en el conjunto de prueba.

        Returns:
            float: La precisión del modelo.

        Raises:
            RuntimeError: Si el modelo no ha sido entrenado aún.
        """
        if self.modelo is None:
            raise RuntimeError("El modelo aún no ha sido entrenado. Llame a 'entrenar_modelo' primero.")

        y_pred = self.modelo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def predecir_perfil(self, nuevos_datos: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones de perfiles de personalidad utilizando el modelo entrenado.

        Args:
            nuevos_datos (pd.DataFrame): DataFrame con nuevos datos de rasgos
                                         (columnas deben coincidir con los rasgos de entrenamiento).

        Returns:
            np.ndarray: Array de las predicciones de perfiles.

        Raises:
            RuntimeError: Si el modelo no ha sido entrenado aún.
            ValueError: Si las columnas de los nuevos datos no coinciden con los rasgos esperados.
        """
        if self.modelo is None:
            raise RuntimeError("El modelo aún no ha sido entrenado. Llame a 'entrenar_modelo' primero.")

        # Asegurarse de que las columnas de los nuevos datos coincidan
        if not all(col in nuevos_datos.columns for col in self.rasgos):
            raise ValueError(f"Las columnas de 'nuevos_datos' deben ser: {', '.join(self.rasgos)}")

        return self.modelo.predict(nuevos_datos[self.rasgos])

    #Comentario 2