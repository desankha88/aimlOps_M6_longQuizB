"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from diabetes_model.config.core import config
from diabetes_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Create a sample input that matches your expectations
    sample_data = {
        'age': [0.038175906, -0.089062939],
        'sex': [0.051680119, -0.044641637],
        'bmi': [0.061696207, -0.011595815],
        'bp': [0.021872386, -0.036656781],
        's1': [-0.044224498, 0.012190569],
        's2': [-0.034820763, 0.024990593],
        's3': [-0.043400846, -0.03613757],
        's4': [-0.002592262, 0.034308859],
        's5': [0.019907486, 0.022687045],
        's6': [-0.017646225, -0.019361911]
    }
    
    X_test = pd.DataFrame(sample_data)
    expected_no_predictions = len(X_test)
    
    # When
    result = make_prediction(input_data=X_test)
    
    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions


#input_data = {'age':[0.038175906, -0.089062939],'sex':[0.051680119, -0.044641637],'bmi':[0.061696207,-0.011595815],'bp':[0.021872386,-0.036656781],'s1':[-0.044224498,0.012190569]
# ,'s2':[-0.034820763,0.024990593],'s3':[-0.043400846,-0.03613757],'s4':[-0.002592262,0.034308859],'s5':[0.019907486,0.022687045],'s6':[-0.017646225,-0.019361911]}


#sample_input_data = pd.DataFrame(input_data)

