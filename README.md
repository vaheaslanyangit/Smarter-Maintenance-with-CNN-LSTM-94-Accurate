# Smarter-Maintenance-with-CNN-LSTM-94-Accurate
This Python script analyzes NASA's CMAPSS dataset to predict aircraft engine RUL. It loads/merges train/test/RUL files, cleans data, computes RUL per cycle, normalizes features, creates 10-step time-series sequences, and trains a CNN-LSTM model (Conv1D, LSTM, Dense). Trained on ~160k rows, eval: train MAE ~22, test ~34.
