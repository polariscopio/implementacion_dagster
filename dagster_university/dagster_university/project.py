from pathlib import Path

from dagster_dbt import DbtProject


dbt_project = DbtProject(
  #project_dir=Path(__file__).joinpath("..", "..", "analytics").resolve(),
  project_dir=Path("C:/Users/pablo/db_postgres").resolve(),
)