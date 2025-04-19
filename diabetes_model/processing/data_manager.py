import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import re
import joblib
import pandas as pd
import typing as t
from sklearn.pipeline import Pipeline

from diabetes_model import __version__ as _version
from diabetes_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

##  Pre-Pipeline Preparation 

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame.drop(labels=config.model_config_.unused_fields, axis=1, inplace=True)

    return data_frame



def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines_old(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            if model_file.is_file():
                model_file.unlink()



def remove_old_pipelines(files_to_keep):
    """
    Remove old model pipelines from the trained_models directory.
    Only deletes files — skips folders like __pycache__.
    """
    model_dir = Path(__file__).resolve().parent.parent / "trained_models"
    
    files_to_keep = files_to_keep + ["__init__.py", ".gitignore"]

    print(f"\n[INFO] Cleaning model directory: {model_dir}")
    print(f"[INFO] Keeping files: {files_to_keep}\n")

    for model_file in model_dir.iterdir():  # Use iterdir instead of glob("*")
        print(f"[DEBUG] Checking: {model_file} | is_file={model_file.is_file()} | is_dir={model_file.is_dir()}")

        # Skip files we want to keep
        if model_file.name in files_to_keep:
            continue

        try:
            if model_file.is_file():
                model_file.unlink()
                print(f"[INFO] Deleted file: {model_file}")
            else:
                print(f"[INFO] Skipping non-file (probably a folder): {model_file}")
        except Exception as e:
            print(f"[ERROR] Could not delete {model_file}: {e}")

