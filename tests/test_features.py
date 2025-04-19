
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from diabetes_model.config.core import config
from diabetes_model.processing.features import convertStrToFloat


def test_transform_converts_strings_to_float(self):
        """Test that transform method converts string columns to float."""
        # Create a sample dataframe with string columns
        data = pd.DataFrame({
            'col1': ['1.1', '2.2', '3.3'],
            'col2': ['4.4', '5.5', '6.6'],
            'col3': ['text1', 'text2', 'text3']  # column not to be transformed
        })
        
        # Initialize and transform
        transformer = convertStrToFloat(variables=["col1", "col2"])
        result = transformer.transform(data)
        
        # Check that specified columns were converted to float
        assert result['col1'].dtype == np.float64
        assert result['col2'].dtype == np.float64
        
        # Check that other columns were not affected
        assert result['col3'].dtype == object
        
        # Check that values were correctly converted
        assert result['col1'].tolist() == [1.1, 2.2, 3.3]
        assert result['col2'].tolist() == [4.4, 5.5, 6.6]
        
        # Check that original dataframe was not modified
        assert data['col1'].dtype == object

def test_transform_with_invalid_values(self):
        """Test transform with values that cannot be converted to float."""
        # Create a sample dataframe with problematic values
        data = pd.DataFrame({
            'col1': ['1.1', 'invalid', '3.3'],
            'col2': ['4.4', '5.5', '6.6']
        })
        
        transformer = convertStrToFloat(variables=["col1", "col2"])
        
        # Should raise ValueError due to invalid conversion
        with pytest.raises(ValueError):
            transformer.transform(data)