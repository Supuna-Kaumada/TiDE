# Migration Guide

This guide is for users who are migrating from the previous version of this TiDE implementation to the new version, which is aligned with the Google Research implementation.

## Key Changes

The new version of the model introduces several key changes to the model architecture and data pipeline.

### Model Architecture

-   **`ResidualBlock`**: The `ResidualBlock` has been refactored to exactly match the Google implementation. The layer ordering is now Dense -> ReLU -> Dense -> Dropout, and the skip connection is applied *before* layer normalization.
-   **`TiDE` Model**: The main `TiDE` model has been significantly refactored. It now includes a `FeatureProjector` for handling covariates, and the `DenseEncoder`, `DenseDecoder`, and `TemporalDecoder` have been updated to align with the original implementation.
-   **Covariate Handling**: The model now explicitly handles past and future covariates.

### Data Pipeline

-   **`load_ett_data`**: The `load_ett_data` function now returns separate dataframes for the target variable, past covariates, future covariates, and time features.
-   **`create_sliding_window`**: The `create_sliding_window` function now takes the target, covariate, and time data as separate arguments and returns past and future feature tensors.

### Training Script

-   **Hyperparameter Management**: The `train.py` script now uses `argparse` for command-line configuration.
-   **Data Loading**: The data loading and preprocessing steps have been updated to work with the new data pipeline.
-   **Model Instantiation**: The model is now instantiated with a more detailed set of parameters, including the number of time features and covariates.

## How to Update Your Code

### 1. Update your `tide/model.py` and `tide/data.py` files

Pull the latest versions of `tide/model.py` and `tide/data.py` from the repository.

### 2. Update your training script

-   **Data Loading**: Modify your data loading code to use the new `load_ett_data` function and `create_sliding_window` function.
-   **Model Instantiation**: Update the instantiation of the `TiDE` model to include the new hyperparameters, such as `num_time_features`, `num_past_covariates`, and `num_future_covariates`.
-   **Training Loop**: Update your training loop to pass the `x_future` tensor to the model's forward pass.
-   **Hyperparameters**: If you were using a custom training script, you will need to update the hyperparameter names to match the new `argparse` arguments in `train.py`.

### Example `train.py` update:

**Old code:**

```python
# Load and prepare data
train_data, test_data, scaler = load_ett_data(zip_file_path, file_name)
X_train, y_train = create_sliding_window(train_data, look_back, horizon)
X_test, y_test = create_sliding_window(test_data, look_back, horizon)

# Reshape data for channel-independent processing
num_features = X_train.shape[2]
X_train = X_train.permute(0, 2, 1).reshape(-1, look_back)
y_train = y_train.permute(0, 2, 1).reshape(-1, horizon)

# ...

# Forward pass
outputs = model(X_batch)
```

**New code:**

```python
# Load and prepare data
(
    train_target, test_target,
    train_covariates, test_covariates,
    train_time, test_time,
    scaler
) = load_ett_data(args.zip_file_path, args.file_name)

X_past_train, X_future_train, y_train = create_sliding_window(
    train_target, train_covariates, train_time, args.look_back, args.horizon
)
# ...

# Forward pass
outputs = model(X_past_batch, X_future_batch)
```
