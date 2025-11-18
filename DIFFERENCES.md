# Differences from the Google Research Implementation

This document outlines the known differences between this PyTorch implementation of TiDE and the original TensorFlow/Keras implementation from Google Research.

## Core Model

-   **Framework**: This implementation is in PyTorch, while the original is in TensorFlow/Keras. This leads to some minor differences in layer implementations and initialization, but the core logic is the same.
-   **Static Covariates**: The original paper mentions static covariates, but they are not included in the public Google Research implementation. This implementation also does not support static covariates.

## Data Pipeline

-   **Data Loading**: The data loading pipeline in this repository is tailored to the ETT datasets and may not be as general as the one in the original implementation. However, it provides the same inputs to the model.

## Training

-   **Optimizer**: The original implementation uses the Adam optimizer with a specific learning rate schedule. This implementation uses the Adam optimizer with a fixed learning rate.
-   **Early Stopping**: The original implementation includes early stopping. This is not yet implemented here.

## Intended Future Alignment

-   Implement a learning rate scheduler to match the original implementation.
-   Add early stopping to the training loop.
-   Generalize the data loading pipeline to support a wider range of datasets.
