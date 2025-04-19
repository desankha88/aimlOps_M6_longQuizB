import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

from diabetes_model.config.core import config
from diabetes_model.pipeline import diabetes_pipe
from diabetes_model.processing.data_manager import load_dataset, save_pipeline
from math import ceil
from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_mean=True)

def transformTarget(y : pd.Series) -> pd.DataFrame:
        y = y.copy()
        sc = StandardScaler(with_mean=True)
        y = sc.fit_transform(y.to_numpy().reshape(-1, 1))
        y = pd.DataFrame(y,columns=['target'])
        #X['target'] = y

        return y  


def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config_.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],  # predictors
        (data[config.model_config_.target_var]),
        test_size=config.model_config_.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config_.random_state1,
    )

    print(X_train.head(2))
    print(y_train.head(2))


    # Pipeline fitting
    diabetes_pipe.fit(X_train,y_train)
    print(X_train.columns)
    pred = diabetes_pipe.predict(X_train)
    print('linear train mse: {}'.format(mean_squared_error(y_train, pred)))
    print('linear train rmse: {}'.format(sqrt(mean_squared_error(y_train, pred))))

    sc = StandardScaler(with_mean=True)
    y_test_scaled = sc.fit(data[config.model_config_.target_var].iloc[ceil(0.9*len(data)):].to_numpy().reshape(-1, 1))
    y_pred = diabetes_pipe.predict(X_test)
    y_pred_inv_scaled = sc.inverse_transform(y_pred.reshape(-1,1))
    print('linear test mse: {}'.format(mean_squared_error(y_test, y_pred)))
    print('linear test rmse: {}'.format(sqrt(mean_squared_error(y_test, y_pred))))
    print("Score (in %):", diabetes_pipe.score(X_test, y_test)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= diabetes_pipe)
    # printing the score
    
if __name__ == "__main__":
    run_training()
