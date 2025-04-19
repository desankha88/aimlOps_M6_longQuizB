import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from diabetes_model.config.core import config
from diabetes_model.processing.features import convertStrToFloat



diabetes_pipe=Pipeline([
     ("convertStrToFloat", convertStrToFloat(variables=config.model_config_.features)),
     ('model_xgboost', xgb.XGBRegressor(n_estimators=config.model_config_.n_estimators,
                                        max_depth=config.model_config_.max_depth,
                                        eta=config.model_config_.eta,
                                        subsample=config.model_config_.subsample, # The fraction of samples to be used for each tree. This is a form of regularization to prevent overfitting.
                                        colsample_bytree=config.model_config_.colsample_bytree, # The fraction of features to be used for each tree. This is another form of regularization.
                                        objective= config.model_config_.objective,# The learning task objective. Here, it indicates a regression task using squared error loss.
                                        random_state=config.model_config_.random_state              # Random seed for reproducibility. Ensures that the results can be replicated across runs.
                                        ))
          
     ])
