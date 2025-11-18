import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_level=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_level)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip_connection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.layer_norm(out + identity)
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_level=0.1):
        super(Encoder, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(ResidualBlock(input_dim if i == 0 else hidden_dim, hidden_dim, hidden_dim, dropout_level))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_level=0.1):
        super(Decoder, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(ResidualBlock(input_dim if i == 0 else hidden_dim, hidden_dim, hidden_dim, dropout_level))
        self.decoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.decoder(x)
        return self.fc(x)

class TemporalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_level=0.1):
        super(TemporalDecoder, self).__init__()
        self.residual_block = ResidualBlock(input_dim, hidden_dim, output_dim, dropout_level)

    def forward(self, x):
        return self.residual_block(x)

class TiDE(nn.Module):
    def __init__(self, look_back, horizon, num_encoder_layers, num_decoder_layers, hidden_dim, decoder_output_dim, temporal_decoder_hidden, dropout_level=0.1):
        super(TiDE, self).__init__()
        self.encoder = Encoder(look_back, hidden_dim, num_encoder_layers, dropout_level)
        self.decoder = Decoder(hidden_dim, hidden_dim, horizon * decoder_output_dim, num_decoder_layers, dropout_level)
        self.temporal_decoder = TemporalDecoder(decoder_output_dim + 1, temporal_decoder_hidden, 1, dropout_level)
        self.decoder_output_dim = decoder_output_dim
        self.horizon = horizon

    def forward(self, x):
        # The input x is expected to have shape (batch_size, look_back)
        batch_size = x.shape[0]

        # Encoder
        encoded = self.encoder(x)

        # Decoder
        decoded = self.decoder(encoded)

        # Reshape for temporal decoder
        decoded = decoded.view(batch_size, self.horizon, self.decoder_output_dim)

        # Since we are not using covariates in this implementation, we will use a placeholder
        # In a real implementation, future covariates would be concatenated here.
        # For now, we'll just use the target series itself as a "covariate"
        # We need to reshape x to match the decoder output
        # Let's take the last value of the look_back window and repeat it for the horizon
        last_val = x[:, -1].unsqueeze(1).repeat(1, self.horizon).unsqueeze(2)

        temporal_input = torch.cat([decoded, last_val], dim=2)

        # Temporal Decoder
        output = self.temporal_decoder(temporal_input)

        return output.squeeze(2)
