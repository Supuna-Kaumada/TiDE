import torch
import numpy as np
import argparse
from tide.data import load_dataset, create_sliding_window
from tide.model import TiDE

def main(args):
    # Load and prepare data
    (
        _, _, test_target,
        _, _, test_covariates,
        _, _, test_time,
        scaler
    ) = load_dataset(args.dataset, target_column='OT' if 'ETT' in args.dataset else 'target')

    X_past_test, X_future_test, y_test = create_sliding_window(
        test_target, test_covariates, test_time, args.look_back, args.horizon
    )

    # Load the trained model
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
        num_time_features=X_past_test.shape[-1] - test_covariates.shape[-1] - 1,
        num_past_covariates=test_covariates.shape[-1],
        num_future_covariates=X_future_test.shape[-1]
    )
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Evaluation
    with torch.no_grad():
        predictions = model(X_past_test, X_future_test)

    # Inverse transform the predictions and the ground truth
    predictions_rescaled = scaler.inverse_transform(predictions.numpy().reshape(-1, 1)).reshape(predictions.shape)
    y_test_rescaled = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).reshape(y_test.shape)

    # Calculate metrics
    mse = np.mean((predictions_rescaled - y_test_rescaled)**2)
    mae = np.mean(np.abs(predictions_rescaled - y_test_rescaled))

    print(f'Test MSE: {mse:.4f}')
    print(f'Test MAE: {mae:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TiDE Evaluation Script')
    parser.add_argument('--dataset', type=str, default='etth1', help='Dataset to use')
    parser.add_argument('--look_back', type=int, default=96)
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_temporal_decoder_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--decoder_output_dim', type=int, default=16)
    parser.add_argument('--temporal_decoder_hidden', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--model_path', type=str, default='tide_model.pth')

    args = parser.parse_args()
    main(args)
