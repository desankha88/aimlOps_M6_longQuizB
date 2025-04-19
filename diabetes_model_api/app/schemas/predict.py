from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[float]

class DataInputSchema(BaseModel):
    age: Optional[float]
    sex: Optional[float]
    bmi: Optional[float]
    bp: Optional[float]
    s1: Optional[float]
    s2: Optional[float]
    s3: Optional[float]
    s4: Optional[float]
    s5: Optional[float]
    s6: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

