from dagster import asset, Output, AssetIn, Int, Float, multi_asset, AssetOut, SourceAsset, IOManager, io_manager, AssetKey
import pandas as pd
from dagster_mlflow import mlflow_tracking
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine


@multi_asset(
    ins={
        "scores_movies_users": AssetIn(
        )
    },
    outs={
        "preprocessed_training_data": AssetOut(),
        "user2Idx": AssetOut(),
        "movie2Idx": AssetOut(),
    }
    
        
)
def preprocessed_data(context, scores_movies_users):
    u_unique = scores_movies_users.user_id.unique()
    user2Idx = {o:i+1 for i,o in enumerate(u_unique)}
    m_unique = scores_movies_users.movie_id.unique()
    movie2Idx = {o:i+1 for i,o in enumerate(m_unique)}
    scores_movies_users.loc[:,'encoded_user_id'] = scores_movies_users.user_id.apply(lambda x: user2Idx[x]).astype(int)
    scores_movies_users.loc[:,'encoded_movie_id'] = scores_movies_users.movie_id.apply(lambda x: movie2Idx[x]).astype(int)
    preprocessed_training_data = scores_movies_users.copy()
    #mlflow = context.resources.mlflow
    #metrics = {"rows: preprocessed_training_data.shape[0]", 
    #           "columns: preprocessed_training_data.columns }
    context.log.info(f"rows processed: {preprocessed_training_data.shape[0]}")
    context.log.info(f"columns processed: {preprocessed_training_data.columns}")
    #preprocessed_training_data.to_csv("C:\\Users\\pablo\\Documents\\PREPROCESED_TRAINING_DATA.csv")
    return preprocessed_training_data, user2Idx, movie2Idx
    


@multi_asset(
    ins={
        "preprocessed_training_data": AssetIn(key=AssetKey("preprocessed_training_data")
        )    
    },
    outs={
        "X_train": AssetOut(),
        "X_test": AssetOut(),
        "y_train": AssetOut(),
        "y_test": AssetOut(),
    }
)

def split_data(context, preprocessed_training_data:pd.DataFrame):
    context.log.info(f"columns in: {preprocessed_training_data.columns}")
    test_size=0.10
    random_state=42
    X = preprocessed_training_data[['user_id', 'movie_id']] #falla con el encoded, encontrar el problema
    y = preprocessed_training_data['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y,        
        test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
    

@multi_asset(
    required_resource_keys={'mlflow'},
    ins={
        "X_train": AssetIn(),
        "y_train": AssetIn(),
        "user2Idx": AssetIn(),
        "movie2Idx": AssetIn(),
    },
    outs={"keras_model": AssetOut(io_manager_key="keras_io_manager")},
    config_schema={
        'batch_size': Int,
        'epochs': Int,
        'learning_rate': Float,
        'embeddings_dim': Int
    }
)
def keras_dot_product_model(context, X_train, y_train, user2Idx, movie2Idx):
    from .model_helper import get_model
    from keras.optimizers import Adam
    mlflow = context.resources.mlflow    
 
    '''
    batch_size = context.op_config["batch_size"]
    epochs = context.op_config["epochs"]
    learning_rate = context.op_config["learning_rate"]
    embeddings_dim = context.op_config["embeddings_dim"]
    '''
    batch_size = 128
    epochs = 10
    learning_rate = 1e-3
    embeddings_dim = 5
    
    model = get_model(len(movie2Idx), len(user2Idx), embeddings_dim)

    #model.compile(Adam(learning_rate=learning_rate), 'mean_squared_error')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    #with mlflow.start_run(run_name="keras_dot_product_model_params"):
    mlflow.log_params(context.op_config)    
    context.log.info(f'batch_size: {batch_size} - epochs: {epochs}')         
   

    
    
    history = model.fit(
        [
            X_train.user_id,
            X_train.movie_id
        ], 
        y_train.rating, 
        batch_size=batch_size,
        epochs=epochs, 
        verbose=1
    )
    for i, l in enumerate(history.history['loss']):
        mlflow.log_metric('mse', l, i)
    context.log.info(f"history keys: {history.history.keys()}")
   

    return model



@multi_asset(
    required_resource_keys={'mlflow'},  # 
    ins={"keras_model": AssetIn()},  # Recibe el modelo desde el asset de entrenamiento
    outs={"model_uri": AssetOut()}
)
def save_model_to_mlflow(context, keras_model):    
    import mlflow.tensorflow
    import tensorflow as tf
    import os.path


    mlflow = context.resources.mlflow  # Obtener la instancia de MLflow

    if not isinstance(keras_model, tf.keras.Model):
        raise TypeError(
            f"Error: keras_dot_product_model debe ser un modelo Keras, pero se recibió {type(keras_dot_product_model)}"
        )

    context.log.info("Guardando el modelo en MLflow...")
    
    tracking_uri = mlflow.get_tracking_uri()
    context.log.info(f" MLflow Tracking URI: {tracking_uri}")

    # Guardar el modelo en MLflow
    with mlflow.start_run(run_name="keras_dot_product_model_save_2",  nested=True):
        logged_model = mlflow.tensorflow.log_model(
            keras_model,
            artifact_path="keras_dot_product_model",
            registered_model_name="keras_dot_product_model"  # Nombre del modelo en MLflow
        )

        context.log.info(f"Modelo guardado en MLflow: {logged_model}")
        context.log.info(f"model uri: {logged_model.model_uri}")
        
    uri = logged_model.model_uri    
    return uri  # Devolver la URI del modelo guardado


@asset(
    required_resource_keys={'mlflow'},
    ins={
        "keras_model": AssetIn(),  # Recibe la URI del modelo desde `save_model_to_mlflow`
        "X_test": AssetIn(),
        "y_test": AssetIn(),
    }
)
def evaluate_model(context, keras_model, X_test:pd.DataFrame, y_test:pd.DataFrame):
    from sklearn.metrics import mean_squared_error

    mlflow = context.resources.mlflow

    # Hacer predicciones con el modelo cargado
    y_pred = keras_model.predict([
        X_test.user_id,
        X_test.movie_id
    ])

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y_pred.reshape(-1), y_test.rating.values)

    # Registrar las métricas en MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_metrics({
            'test_mse': mse,
            'test_rmse': mse**(0.5)
        })

    context.log.info(f"Evaluación completada - MSE: {mse}, RMSE: {mse**(0.5)}")



'''
@asset(
    required_resource_keys={'mlflow'},
    ins={
        "model_uri": AssetIn(key=AssetKey("model_uri")),  # Recibe la URI del modelo desde `save_model_to_mlflow`
        "X_test": AssetIn(),
        "y_test": AssetIn(),
    }
)
def evaluate_model(context, model_uri:str, X_test:pd.DataFrame, y_test:pd.DataFrame):
    import mlflow.pyfunc
    from sklearn.metrics import mean_squared_error

    mlflow = context.resources.mlflow

    context.log.info(f"Cargando modelo desde: {model_uri}")

    # Cargar el modelo desde MLflow
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Hacer predicciones con el modelo cargado
    y_pred = loaded_model.predict([
        X_test.user_id,
        X_test.movie_id
    ])

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y_pred.reshape(-1), y_test.rating.values)

    # Registrar las métricas en MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_metrics({
            'test_mse': mse,
            'test_rmse': mse**(0.5)
        })

    context.log.info(f"Evaluación completada - MSE: {mse}, RMSE: {mse**(0.5)}")
'''
    
