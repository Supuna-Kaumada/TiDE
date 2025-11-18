import torch
import unittest
from tide.model import ResidualBlock, TiDE
from tide.data import load_dataset
import os

class TestData(unittest.TestCase):
    def setUp(self):
        """
        Set up a dummy dataset for testing.
        """
        self.dataset_name = 'dummy'
        self.data_path = 'tests/dummy_data.csv'
        from tide.data import DATA_DICT
        DATA_DICT[self.dataset_name] = {
            'boundaries': [10, 13, 16],
            'data_path': self.data_path,
            'freq': 'H',
        }

    def test_load_dataset(self):
        """
        Tests that the load_dataset function correctly loads and processes data.
        """
        (
            train_target, val_target, test_target,
            train_covariates, val_covariates, test_covariates,
            train_time, val_time, test_time,
            scaler
        ) = load_dataset(self.dataset_name, target_column='target')

        self.assertEqual(train_target.shape, (10, 1))
        self.assertEqual(val_target.shape, (3, 1))
        self.assertEqual(test_target.shape, (3, 1))
        self.assertEqual(train_covariates.shape, (10, 2))
        self.assertEqual(val_covariates.shape, (3, 2))
        self.assertEqual(test_covariates.shape, (3, 2))
        self.assertEqual(train_time.shape, (10, 4))
        self.assertEqual(val_time.shape, (3, 4))
        self.assertEqual(test_time.shape, (3, 4))


class TestResidualBlock(unittest.TestCase):
    def test_forward_pass_shape(self):
        """
        Tests that the forward pass of the ResidualBlock returns the correct shape.
        """
        input_dim = 64
        hidden_dim = 128
        output_dim = 64
        batch_size = 32

        model = ResidualBlock(input_dim, hidden_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, output_dim))

    def test_projection(self):
        """
        Tests that the projection is applied when input and output dimensions differ.
        """
        input_dim = 64
        hidden_dim = 128
        output_dim = 32
        batch_size = 32

        model = ResidualBlock(input_dim, hidden_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsNotNone(model.projector)


class TestTiDE(unittest.TestCase):
    def test_forward_pass_shape(self):
        """
        Tests that the forward pass of the TiDE model returns the correct shape.
        """
        batch_size = 32
        look_back = 96
        horizon = 48
        num_past_covariates = 6
        num_future_covariates = 2

        model = TiDE(
            lookback_len=look_back,
            horizon=horizon,
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=128,
            decoder_output_dim=16,
            temporal_decoder_hidden=64,
            num_temporal_decoder_layers=1,
            num_past_covariates=num_past_covariates,
            num_future_covariates=num_future_covariates,
        )

        x_past = torch.randn(batch_size, look_back, 1 + num_past_covariates)
        x_future = torch.randn(batch_size, horizon, num_future_covariates)
        output = model(x_past, x_future)

        self.assertEqual(output.shape, (batch_size, horizon))

    def test_forward_pass_no_covariates(self):
        """
        Tests that the forward pass of the TiDE model works without covariates.
        """
        batch_size = 32
        look_back = 96
        horizon = 48

        model = TiDE(
            lookback_len=look_back,
            horizon=horizon,
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=128,
            decoder_output_dim=16,
            temporal_decoder_hidden=64,
            num_temporal_decoder_layers=1,
        )

        x_past = torch.randn(batch_size, look_back, 1)
        output = model(x_past)

        self.assertEqual(output.shape, (batch_size, horizon))

if __name__ == '__main__':
    unittest.main()
