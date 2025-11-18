import torch
import torch.nn as nn
import torch.optim as optim
from tide.data import load_ett_data, create_sliding_window
from tide.model import TiDE
import argparse

def main(args):
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
    X_past_test, X_future_test, y_test = create_sliding_window(
        test_target, test_covariates, test_time, args.look_back, args.horizon
    )

    # Instantiate model, loss function, and optimizer
    model = TiDE(
        lookback_len=args.look_back,
        horizon=args.horizon,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        hidden_dim=args.hidden_dim,
        decoder_output_dim=args.decoder_output_dim,
        temporal_decoder_hidden=args.temporal_decoder_hidden,
        num_temporal_decoder_layers=args.num_temporal_decoder_layers,
        dropout_rate=args.dropout_rate,
        use_layer_norm=args.use_layer_norm,
        num_time_features=X_past_train.shape[-1] - train_covariates.shape[-1] - 1, # time features
        num_past_covariates=train_covariates.shape[-1],
        num_future_covariates=X_future_train.shape[-1]
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        for i in range(0, len(X_past_train), args.batch_size):
            # Get batch
            X_past_batch = X_past_train[i:i+args.batch_size]
            X_future_batch = X_future_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]

            # Forward pass
            outputs = model(X_past_batch, X_future_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}')

        # Validation
        with torch.no_grad():
            val_outputs = model(X_past_test, X_future_test)
            val_loss = criterion(val_outputs, y_test)
            print(f'Validation Loss: {val_loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'tide_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TiDE Training Script')
    parser.add_argument('--zip_file_path', type=str, default='datasets.zip')
    parser.add_argument('--file_name', type=str, default='datasets/ETT-small/ETTh1.csv')
    parser.add_argument('--look_back', type=int, default=96)
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_temporal_decoder_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--decoder_output_dim', type=int, default=16)
    parser.add_argument('--temporal_decoder_hidden', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_layer_norm', action='store_true')

    args = parser.parse_args()
    main(args)
