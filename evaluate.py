import torch
from tide.data import load_ett_data, create_sliding_window
from tide.model import TiDE
import numpy as np

# Hyperparameters
zip_file_path = 'datasets.zip'
file_name = 'datasets/ETT-small/ETTh1.csv'
look_back = 96
horizon = 96
num_encoder_layers = 2
num_decoder_layers = 2
hidden_dim = 128
decoder_output_dim = 16
temporal_decoder_hidden = 64
dropout_level = 0.1
model_path = 'tide_model.pth'

# Load and prepare data
_, test_data, scaler = load_ett_data(zip_file_path, file_name)
X_test, y_test = create_sliding_window(test_data, look_back, horizon)

# Reshape data for channel-independent processing
num_features = X_test.shape[2]
X_test = X_test.permute(0, 2, 1).reshape(-1, look_back)
y_test = y_test.permute(0, 2, 1).reshape(-1, horizon)

# Load the trained model
model = TiDE(look_back, horizon, num_encoder_layers, num_decoder_layers, hidden_dim, decoder_output_dim, temporal_decoder_hidden, dropout_level)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluation
with torch.no_grad():
    predictions = model(X_test)

# Inverse transform the predictions and the ground truth
predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, num_features, horizon).reshape(-1, num_features))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, num_features, horizon).reshape(-1, num_features))

# Calculate metrics
mse = np.mean((predictions_rescaled - y_test_rescaled)**2)
mae = np.mean(np.abs(predictions_rescaled - y_test_rescaled))

print(f'Test MSE: {mse:.4f}')
print(f'Test MAE: {mae:.4f}')
