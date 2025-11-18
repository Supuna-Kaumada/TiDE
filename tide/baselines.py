import numpy as np
from sklearn.linear_model import LinearRegression

def naive_persistence(last_observation, horizon):
    """
    A naive persistence model that repeats the last observation for the entire horizon.

    Args:
        last_observation (float): The last observed value.
        horizon (int): The number of time steps to predict.

    Returns:
        np.array: An array of predictions.
    """
    return np.full(horizon, last_observation)

def linear_regression_model(X_train, y_train, X_test):
    """
    A simple linear regression model.

    Args:
        X_train (np.array): The training input data.
        y_train (np.array): The training output data.
        X_test (np.array): The test input data.

    Returns:
        np.array: The predictions for the test data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)
