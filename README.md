# TiDE: Time-series Dense Encoder

This is a PyTorch implementation of the TiDE model from the paper "Long-term Forecasting with TiDE: Time-series Dense Encoder" (arXiv:2304.08424).

This implementation has been refactored to align with the original Google Research implementation [here](https://github.com/google-research/google-research/tree/master/tide).

## Documentation

-   **[ARCHITECTURE.md](ARCHITECTURE.md)**: A detailed explanation of the model architecture.
-   **[DIFFERENCES.md](DIFFERENCES.md)**: A document outlining the differences between this implementation and the original.
-   **[MIGRATION.md](MIGRATION.md)**: A guide for migrating from the previous version of this implementation.

## Project Structure

-   `tide/`: This directory contains the core source code for the TiDE model.
    -   `data.py`: Data loading and preprocessing utilities.
    -   `model.py`: The TiDE model architecture.
-   `train.py`: The training script.
-   `evaluate.py`: The evaluation script.
-   `requirements.txt`: The project dependencies.
-   `datasets.zip`: The ETT datasets.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Unzip the datasets:**

    Due to file size limitations, you will need to unzip the ETT datasets individually.

    ```bash
    unzip datasets.zip 'datasets/ETT-small/ETTh1.csv'
    unzip datasets.zip 'datasets/ETT-small/ETTh2.csv'
    ```

    *Note: The larger ETTm files are not supported in this implementation due to file size constraints.*

## Training

To train the TiDE model, run the `train.py` script with your desired hyperparameters:

```bash
python train.py --num_epochs 20 --learning_rate 0.0005
```

This will train the model on the ETTh1 dataset and save the trained model to `tide_model.pth`. For a full list of configurable hyperparameters, see the `argparse` section in `train.py`.

## Evaluation

To evaluate the trained model, run the `evaluate.py` script:

```bash
python evaluate.py
```

This will load the trained model and evaluate its performance on the test set, printing the MSE and MAE.
