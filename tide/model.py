import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout_rate: float = 0.0,
                 use_layer_norm: bool = False):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._dropout_rate = dropout_rate
        self._use_layer_norm = use_layer_norm

        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        if self._use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        if input_dim != output_dim:
            self.projector = nn.Linear(input_dim, output_dim)
        else:
            self.projector = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block forward pass."""
        identity = x

        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.dropout(x)

        if self.projector is not None:
            identity = self.projector(identity)

        x += identity

        if self._use_layer_norm:
            x = self.layer_norm(x)
        return x

class FeatureProjector(nn.Module):
    """Projects features to a new dimension."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects features to a new dimension."""
        return self.dense(x)


class DenseEncoder(nn.Module):
    """A dense encoder."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float = 0.0,
                 use_layer_norm: bool = False):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                ResidualBlock(
                    input_dim=input_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    use_layer_norm=use_layer_norm))
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense encoder forward pass."""
        x = self.encoder(x)
        return self.projection(x)


class DenseDecoder(nn.Module):
    """A dense decoder."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float = 0.0,
                 use_layer_norm: bool = False):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                ResidualBlock(
                    input_dim=input_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim if i < num_layers - 1 else output_dim,
                    dropout_rate=dropout_rate,
                    use_layer_norm=use_layer_norm))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense decoder forward pass."""
        return self.decoder(x)


class TemporalDecoder(nn.Module):
    """A temporal decoder."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout_rate: float = 0.0,
                 use_layer_norm: bool = False):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                ResidualBlock(
                    input_dim=input_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    use_layer_norm=use_layer_norm))
        self.decoder = nn.Sequential(*layers)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal decoder forward pass."""
        x = self.decoder(x)
        return self.projection(x)


class TiDE(nn.Module):
    """TiDE model."""

    def __init__(self,
                 lookback_len: int,
                 horizon: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 hidden_dim: int,
                 decoder_output_dim: int,
                 temporal_decoder_hidden: int,
                 num_temporal_decoder_layers: int,
                 dropout_rate: float = 0.0,
                 use_layer_norm: bool = False,
                 num_time_features: int = 0,
                 num_past_covariates: int = 0,
                 num_future_covariates: int = 0):
        super().__init__()
        self.lookback_len = lookback_len
        self.horizon = horizon
        self.decoder_output_dim = decoder_output_dim

        self.encoder = DenseEncoder(
            input_dim=lookback_len * (1 + num_past_covariates + num_time_features),
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_encoder_layers,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm)

        self.decoder = DenseDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=horizon * decoder_output_dim,
            num_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm)

        self.temporal_decoder = TemporalDecoder(
            input_dim=decoder_output_dim + num_future_covariates,
            hidden_dim=temporal_decoder_hidden,
            output_dim=1,
            num_layers=num_temporal_decoder_layers,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm)

    def forward(self,
                x_past: torch.Tensor,
                x_future: torch.Tensor = None) -> torch.Tensor:
        """TiDE forward pass."""
        batch_size = x_past.shape[0]

        # Flatten the input
        x_past_flat = x_past.view(batch_size, -1)

        # Encoder
        encoded = self.encoder(x_past_flat)

        # Decoder
        decoded = self.decoder(encoded)

        # Reshape for temporal decoder
        decoded = decoded.view(batch_size, self.horizon, self.decoder_output_dim)

        if x_future is not None:
            temporal_input = torch.cat([decoded, x_future], dim=-1)
        else:
            temporal_input = decoded

        # Temporal Decoder
        output = self.temporal_decoder(temporal_input)

        return output.squeeze(-1)
