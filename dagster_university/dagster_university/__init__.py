from dagster import Definitions, load_assets_from_modules, \
load_assets_from_package_module, define_asset_job, \
AssetSelection, ScheduleDefinition, FilesystemIOManager, SourceAsset, \
IOManager, io_manager, resource

from dagster_mlflow import mlflow_tracking


#airbyte modules
from dagster_airbyte import AirbyteResource, load_assets_from_airbyte_instance  #esto es del tutorial de airbyte
from dagster import AssetKey

#modulos locales
from .assets import dbt, train_model, model_helper # Import the dbt assets

#dbt
from .resources import  dbt_resource # import the dbt resource
from .resources import  dbt_io_manager   # import the iomanager resource
from .resources import  keras_model_io_manager

# ... existing calls to load_assets_from_modules
dbt_analytics_assets = load_assets_from_modules(modules=[dbt]) # Load the assets from the file
recommender_train = load_assets_from_modules(modules=[train_model]) # Load the assets from the file
recommender_helper = load_assets_from_modules(modules=[model_helper]) # Load the assets from the file




# Función para generar claves únicas
def connection_to_asset_key_fn(connection_name, stream_name) -> AssetKey:
    # Verifica si connection_name es un objeto AirbyteConnectionMetadata
    if hasattr(connection_name, "name"):
        connection_name = connection_name.name  # Extrae el nombre como string

    # Asegúrate de que connection_name y stream_name sean strings
    if not isinstance(connection_name, str) or not isinstance(stream_name, str):
        raise ValueError(
            f"connection_name y stream_name deben ser strings. Recibido: {type(connection_name)}, {type(stream_name)}"
        )

    # Devuelve un AssetKey válido
    return AssetKey([connection_name, stream_name])

# Load all assets from your Airbyte instance 
airbyte_assets = load_assets_from_airbyte_instance(
    # Connect to your OSS Airbyte instance
    AirbyteResource(
        host="localhost",
        port="8000",
        # If using basic auth, include username and password:
        username="pablofp92@gmail.com",
        password= "10gcAIlNYP3ZqOHs89NaKmqORhoiZPpN",
    ),
    connection_to_asset_key_fn=connection_to_asset_key_fn
)


#configs
'''
training_config = {
    'keras_dot_product_model': {
        'config': {
            'batch_size': 128,
            'epochs': 10,
            'learning_rate': 1e-3,
            'embeddings_dim': 5
        }
    }
}
'''
mlflow_resources = {
    "experiment_name": "recommender_system_3",
    "mlflow_tracking_uri": "http://localhost:5000",
    #"default_artifact_root": "file:///C:/Users/pablo/Documents/mlops-ecosystem-main/mlflow_data"  

}

# Configuración específica para keras_dot_product_model
op_config = {
    "ops": {
        "keras_dot_product_model": {
            "config": {
                "batch_size": 128,
                "embeddings_dim": 5,
                "epochs": 10,
                "learning_rate": 0.001
            }
        }
    }
}

# Crear un job para los assets relacionados
keras_job = define_asset_job(
    name="keras_job", 
    selection=["keras_dot_product_model"]  # Incluimos solo los assets necesarios
)



'''
job_training_config = {
    'resources': {
        **mlflow_resources
    },
    'ops': {
        **training_config
    }
}

job_all_config = {
    'resources': {
        **mlflow_resources
    },
    'ops': {
        **training_config
    }
}
'''
#jobs


@resource
def dummy_mlflow_resource(_):
    pass

# ... other declarations

defs = Definitions(
    assets=[*dbt_analytics_assets, airbyte_assets, *recommender_train, *recommender_helper], # Add the dbt assets to your code location
    #jobs = [define_asset_job("full_process",config=job_all_config)],
    #jobs=[keras_job],
    resources={
        "dbt": dbt_resource,
        "io_manager": dbt_io_manager,
        "mlflow": mlflow_tracking.configured(mlflow_resources),
        "keras_io_manager": keras_model_io_manager,
    }
    
  # .. other definitions
)

defs.default_job_config = op_config
