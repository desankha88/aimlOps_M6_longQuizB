import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from diabetes_model import __version__ as _version
from diabetes_model.config.core import config
from diabetes_model.pipeline import diabetes_pipe
from diabetes_model.processing.data_manager import load_pipeline
#from diabetes_model.processing.data_manager import pre_pipeline_preparation
from diabetes_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
diabetes_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    print("After validation:", validated_data.columns) 

    #validated_data = pre_pipeline_preparation(data_frame=validated_data)
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = diabetes_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = diabetes_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'age':[0.0344433679824036],'sex':[0.0506801187398186],'bmi':[0.0476846495582328],'bp':[-0.0469846340029485],'s1':[-0.0125765826858202],
                's2':[0.0168487333575743],'s3':[-0.0802172236928943],'s4':[0.130251773155092],'s5':[0.0734069578883319],'s6':[0.0859065477110576]}
    
    make_prediction(input_data=data_in)
