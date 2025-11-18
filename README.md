# TiDE: Time-series Dense Encoder

This is a PyTorch implementation of the TiDE model from the paper "Long-term Forecasting with TiDE: Time-series Dense Encoder" (arXiv:2304.08424).

## Project Structure

- `tide/`: This directory contains the core source code for the TiDE model.
  - `data.py`: Data loading and preprocessing utilities.
  - `model.py`: The TiDE model architecture.
  - `baselines.py`: Baseline models for comparison.
- `train.py`: The training script.
- `evaluate.py`: The evaluation script.
- `requirements.txt`: The project dependencies.
- `datasets.zip`: The ETT datasets.

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

To train the TiDE model, run the `train.py` script:

```bash
python train.py
```

This will train the model on the ETTh1 dataset and save the trained model to `tide_model.pth`.

## Evaluation

To evaluate the trained model, run the `evaluate.py` script:

```bash
python evaluate.py
```

This will load the trained model and evaluate its performance on the test set, printing the MSE and MAE.
