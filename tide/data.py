import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

DATA_DICT = {
    'ettm2': {
        'boundaries': [34560, 46080, 57600],
        'data_path': './datasets/ETT-small/ETTm2.csv',
        'freq': '15min',
    },
    'ettm1': {
        'boundaries': [34560, 46080, 57600],
        'data_path': './datasets/ETT-small/ETTm1.csv',
        'freq': '15min',
    },
    'etth2': {
        'boundaries': [8640, 11520, 14400],
        'data_path': './datasets/ETT-small/ETTh2.csv',
        'freq': 'H',
    },
    'etth1': {
        'boundaries': [8640, 11520, 14400],
        'data_path': './datasets/ETT-small/ETTh1.csv',
        'freq': 'H',
    },
    'elec': {
        'boundaries': [18413, 21044, 26304],
        'data_path': './datasets/electricity/electricity.csv',
        'freq': 'H',
    },
    'traffic': {
        'boundaries': [12280, 14036, 17544],
        'data_path': './datasets/traffic/traffic.csv',
        'freq': 'H',
    },
    'weather': {
        'boundaries': [36887, 42157, 52696],
        'data_path': './datasets/weather/weather.csv',
        'freq': '10min',
    },
}


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

def load_dataset(dataset_name, target_column='OT'):
    """
    Loads a dataset and prepares it for the TiDE model.
    """
    dataset_info = DATA_DICT[dataset_name]
    df = pd.read_csv(dataset_info['data_path'])

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Time features
    time_features = get_time_features(df.index)

    # Target and covariates
    target_df = df[[target_column]]
    covariate_df = df.drop(columns=[target_column])

    # Train/val/test split
    train_end, val_end, test_end = dataset_info['boundaries']

    train_target = target_df[:train_end]
    val_target = target_df[train_end:val_end]
    test_target = target_df[val_end:test_end]

    train_covariates = covariate_df[:train_end]
    val_covariates = covariate_df[train_end:val_end]
    test_covariates = covariate_df[val_end:test_end]

    train_time_features = time_features[:train_end]
    val_time_features = time_features[train_end:val_end]
    test_time_features = time_features[val_end:test_end]

    # Scaling
    scaler_target = StandardScaler()
    train_target_scaled = scaler_target.fit_transform(train_target)
    val_target_scaled = scaler_target.transform(val_target)
    test_target_scaled = scaler_target.transform(test_target)

    scaler_covariate = StandardScaler()
    train_covariates_scaled = scaler_covariate.fit_transform(train_covariates)
    val_covariates_scaled = scaler_covariate.transform(val_covariates)
    test_covariates_scaled = scaler_covariate.transform(test_covariates)

    scaler_time = StandardScaler()
    train_time_features_scaled = scaler_time.fit_transform(train_time_features)
    val_time_features_scaled = scaler_time.transform(val_time_features)
    test_time_features_scaled = scaler_time.transform(test_time_features)

    return (
        train_target_scaled, val_target_scaled, test_target_scaled,
        train_covariates_scaled, val_covariates_scaled, test_covariates_scaled,
        train_time_features_scaled, val_time_features_scaled, test_time_features_scaled,
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
