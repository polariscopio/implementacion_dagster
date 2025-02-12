from dagster_dbt import DbtCliResource
from dagster import asset, AssetIn, Int, Float, multi_asset, AssetOut, SourceAsset, IOManager, io_manager
from sqlalchemy import create_engine
import pandas as pd
import mlflow.tensorflow
import tensorflow as tf
import os



from ..project import dbt_project
# the import lines go at the top of the file

# this can be defined anywhere below the imports
dbt_resource = DbtCliResource(
    project_dir=dbt_project,
)

class DbtIOManager(IOManager):
    def __init__(self, connection_string):
        self.connection_string = connection_string

    def load_input(self, context):
        # Obtener el nombre del asset (modelo dbt)
        dbt_model_name = "scores_movies_users"
        engine = create_engine(self.connection_string)

        # Leer la tabla directamente desde la base de datos
        query = f"SELECT * FROM target.{dbt_model_name}"
        with engine.connect() as connection:
            return pd.read_sql(query, connection)

    def handle_output(self, context, obj):
        # No necesitamos manejar outputs aquí porque dbt ya lo gestiona
        pass

@io_manager
def dbt_io_manager():
    return DbtIOManager("postgresql://postgres:mysecretpassword@localhost:5432/mlops")



class KerasModelIOManager(IOManager):
    def handle_output(self, context, obj):
        """Guarda el modelo Keras en un archivo local y en MLflow."""
        model_dir = "dagster_models"
        os.makedirs(model_dir, exist_ok=True)  # Crear carpeta si no existe

        model_path = os.path.join(model_dir, "keras_dot_product_model.keras")
        obj.save(model_path)  # Guardar modelo en disco

        context.log.info(f"Modelo guardado localmente en {model_path}")

        # Guardar en MLflow
        #mlflow.tensorflow.log_model(obj, artifact_path="file:///C://Users//pablo/Documents//mlops-ecosystem-main//mlflow_data/keras_dot_product_model")
        #context.log.info("Modelo guardado en MLflow")

    def load_input(self, context):
        """Carga el modelo Keras desde el archivo local."""
        model_path = "dagster_models/keras_dot_product_model.keras"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo Keras no se encontró en {model_path}")

        context.log.info(f"Cargando modelo desde {model_path}")
        return tf.keras.models.load_model(model_path)  # Cargar modelo desde el archivo

# Registrar el IOManager en Dagster
@io_manager
def keras_model_io_manager(_):
    return KerasModelIOManager()