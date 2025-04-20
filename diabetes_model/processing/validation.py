import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from pydantic import ConfigDict
from diabetes_model.config.core import config
from diabetes_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        print("Validation error: ", error.json())
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    age: Optional[float]
    bmi: Optional[float]
    bp: Optional[float]
    s1: Optional[float]
    s2: Optional[float]
    s3: Optional[float]
    s4: Optional[float]
    s5: Optional[float]
    s6: Optional[float]
    
    # Make sex truly optional by giving it a default value
    sex: Optional[float] = None

    model_config = ConfigDict(
        # config options here
        extra = "ignore"  # Allows extra fields not in the model
    )
    
    # Add model configuration to allow extra fields
    #class Config:
    #    extra = "ignore"  # Allows extra fields not in the model


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
