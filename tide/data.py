import pandas as pd
import torch
from zipfile import ZipFile
from sklearn.preprocessing import StandardScaler

def load_ett_data(zip_file_path, file_name, test_size=0.2):
    """
    Loads an ETT dataset from a zip file.

    Args:
        zip_file_path (str): The path to the zip file.
        file_name (str): The name of the CSV file in the zip file.
        test_size (float): The proportion of the dataset to use for testing.

    Returns:
        tuple: A tuple containing the training and testing dataframes.
    """
    with ZipFile(zip_file_path) as zf:
        with zf.open(file_name) as f:
            df = pd.read_csv(f)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    train_size = int(len(df) * (1 - test_size))
    train_df = df[:train_size]
    test_df = df[train_size:]

    scaler = StandardScaler()
    train_df_scaled = scaler.fit_transform(train_df)
    test_df_scaled = scaler.transform(test_df)

    return train_df_scaled, test_df_scaled, scaler

def create_sliding_window(data, look_back, horizon):
    """
    Creates a sliding window dataset.

    Args:
        data (np.array): The input data.
        look_back (int): The number of time steps to look back.
        horizon (int): The number of time steps to predict.

    Returns:
        tuple: A tuple containing the input and output tensors.
    """
    X, Y = [], []
    for i in range(len(data) - look_back - horizon + 1):
        X.append(data[i:(i + look_back)])
        Y.append(data[(i + look_back):(i + look_back + horizon)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
