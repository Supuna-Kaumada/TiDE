import pandas as pd
import torch
from zipfile import ZipFile
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_time_features(dt):
    """
    Generates time features from a pandas DatetimeIndex.
    """
    features = {
        "month": dt.month,
        "day": dt.day,
        "weekday": dt.weekday,
        "hour": dt.hour,
    }
    return pd.DataFrame(features, index=dt)

def load_ett_data(zip_file_path, file_name, target_column='OT', test_size=0.2):
    """
    Loads a dataset from a zip file and prepares it for the TiDE model.

    Args:
        zip_file_path (str): The path to the zip file.
        file_name (str): The name of the CSV file in the zip file.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to use for testing.

    Returns:
        tuple: A tuple containing training and testing dataframes for target,
               past covariates, future covariates, and time features, plus the scaler.
    """
    with ZipFile(zip_file_path) as zf:
        with zf.open(file_name) as f:
            df = pd.read_csv(f)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Time features
    time_features = get_time_features(df.index)

    # Target and covariates
    target_df = df[[target_column]]
    covariate_df = df.drop(columns=[target_column])

    # Train/test split
    train_size = int(len(df) * (1 - test_size))

    train_target = target_df[:train_size]
    test_target = target_df[train_size:]

    train_covariates = covariate_df[:train_size]
    test_covariates = covariate_df[train_size:]

    train_time_features = time_features[:train_size]
    test_time_features = time_features[train_size:]

    # Scaling
    scaler_target = StandardScaler()
    train_target_scaled = scaler_target.fit_transform(train_target)
    test_target_scaled = scaler_target.transform(test_target)

    scaler_covariate = StandardScaler()
    train_covariates_scaled = scaler_covariate.fit_transform(train_covariates)
    test_covariates_scaled = scaler_covariate.transform(test_covariates)

    scaler_time = StandardScaler()
    train_time_features_scaled = scaler_time.fit_transform(train_time_features)
    test_time_features_scaled = scaler_time.transform(test_time_features)

    return (
        train_target_scaled, test_target_scaled,
        train_covariates_scaled, test_covariates_scaled,
        train_time_features_scaled, test_time_features_scaled,
        scaler_target
    )

def create_sliding_window(target_data, covariate_data, time_data, look_back, horizon):
    """
    Creates a sliding window dataset for the TiDE model.
    """
    X_past, X_future, Y = [], [], []

    for i in range(len(target_data) - look_back - horizon + 1):
        # Past data
        past_target = target_data[i:(i + look_back)]
        past_covariates = covariate_data[i:(i + look_back)]
        past_time = time_data[i:(i + look_back)]

        # Future data
        future_target = target_data[(i + look_back):(i + look_back + horizon)]
        future_covariates = covariate_data[(i + look_back):(i + look_back + horizon)]
        future_time = time_data[(i + look_back):(i + look_back + horizon)]

        # Combine past features
        x_past = np.concatenate([past_target, past_covariates, past_time], axis=1)

        # Combine future features
        x_future = np.concatenate([future_covariates, future_time], axis=1)

        X_past.append(x_past)
        X_future.append(x_future)
        Y.append(future_target)

    return (
        torch.tensor(np.array(X_past), dtype=torch.float32),
        torch.tensor(np.array(X_future), dtype=torch.float32),
        torch.tensor(np.array(Y), dtype=torch.float32).squeeze(-1)
    )
