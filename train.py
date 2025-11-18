import torch
import torch.nn as nn
import torch.optim as optim
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
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Load and prepare data
train_data, test_data, scaler = load_ett_data(zip_file_path, file_name)
X_train, y_train = create_sliding_window(train_data, look_back, horizon)
X_test, y_test = create_sliding_window(test_data, look_back, horizon)

# Reshape data for channel-independent processing
num_features = X_train.shape[2]
X_train = X_train.permute(0, 2, 1).reshape(-1, look_back)
y_train = y_train.permute(0, 2, 1).reshape(-1, horizon)

X_test = X_test.permute(0, 2, 1).reshape(-1, look_back)
y_test = y_test.permute(0, 2, 1).reshape(-1, horizon)


# Instantiate model, loss function, and optimizer
model = TiDE(look_back, horizon, num_encoder_layers, num_decoder_layers, hidden_dim, decoder_output_dim, temporal_decoder_hidden, dropout_level)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        # Get batch
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Validation
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        print(f'Validation Loss: {val_loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'tide_model.pth')
