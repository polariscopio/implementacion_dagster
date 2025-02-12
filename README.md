# Pasos previos

Tener corriendo mlflow y airbyte (ver repositorio anterior)  




# Configuración del Entorno

Este proyecto utiliza **Poetry** para la gestión de dependencias y entornos virtuales. .

## 🔧 Activar el Entorno Virtual

```sh
poetry env activate
poetry install
```

O instalar cada librería por separado si da problemas

```sh
poetry env activate
poetry add numpy
poetry add mlflow
poetry add dagster
poetry add dagster-mlflow
poetry add tensorflow-io-gcs-filesystem
poetry add tensorflow
poetry add dagster-airbyte
poetry add dagster-dbt
poetry add dagster-webserver
poetry add requests
poetry add pandas
```

# Iniciar Dagster

```sh
cd dagster_university
dagster dev --use-legacy-code-server-behavior
```


