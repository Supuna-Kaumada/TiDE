# TiDE Architecture

This document provides a detailed overview of the TiDE (Time-series Dense Encoder) model architecture, as implemented in this repository. The architecture is designed to align with the original Google Research implementation.

## Core Components

The TiDE model is composed of four main components:

1.  **Dense Encoder**: A stack of `ResidualBlock`s that processes the flattened input of historical data and past covariates.
2.  **Dense Decoder**: Another stack of `ResidualBlock`s that takes the encoded representation and produces an intermediate representation.
3.  **Temporal Decoder**: A final decoder that processes the intermediate representation and future covariates on a per-timestep basis to produce the final predictions.
4.  **Feature Projector**: A simple linear layer used to project features to a new dimension, used for handling time-varying covariates.

## Residual Block

The `ResidualBlock` is the core building block of the TiDE model. It consists of two dense layers with a ReLU activation in between, a dropout layer, and a skip connection. Layer normalization is optional.

The forward pass of the `ResidualBlock` is as follows:

1.  The input is passed through a dense layer.
2.  A ReLU activation is applied.
3.  The result is passed through another dense layer.
4.  Dropout is applied.
5.  The result is added to the original input (skip connection).
6.  If enabled, layer normalization is applied.

## Data Flow

The following diagram illustrates the data flow through the TiDE model:

```
+---------------------+
|      Input Data     |
| (Past + Covariates) |
+---------------------+
           |
           v
+---------------------+
|    Dense Encoder    |
+---------------------+
           |
           v
+---------------------+
|    Dense Decoder    |
+---------------------+
           |
           v
+---------------------+
|  Temporal Decoder   |
| (with Future Covs)  |
+---------------------+
           |
           v
+---------------------+
|       Output        |
|    (Predictions)    |
+---------------------+
```

## Detailed Architecture

### 1. Input Processing

-   **Historical Data**: The historical time series data (`x_past`).
-   **Past Covariates**: Time-varying covariates that are known in the past.
-   **Future Covariates**: Time-varying covariates that are known in the future.
-   **Static Covariates**: Not currently supported, in alignment with the original Google Research implementation.

The past data and past covariates are flattened and concatenated before being fed into the Dense Encoder.

### 2. Dense Encoder

The Dense Encoder takes the flattened input and passes it through a series of `ResidualBlock`s. The output of the encoder is a latent representation of the input data.

### 3. Dense Decoder

The Dense Decoder takes the latent representation from the encoder and passes it through another series of `ResidualBlock`s. The output of the decoder is an intermediate representation that is then reshaped to have a temporal dimension.

### 4. Temporal Decoder

The Temporal Decoder processes the reshaped output from the Dense Decoder, along with the future covariates, on a per-timestep basis. This allows the model to incorporate future information into its predictions. The output of the Temporal Decoder is the final prediction for each timestep in the horizon.
