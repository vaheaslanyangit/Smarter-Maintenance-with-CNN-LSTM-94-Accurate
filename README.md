# NASA CMAPSS Engine RUL Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

This repository contains a Python script for predictive maintenance on aircraft engines using NASA's Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset. The script processes sensor data to predict the Remaining Useful Life (RUL) of engines, employing data preprocessing, feature normalization, time-series sequence generation, and a hybrid Convolutional Neural Network (CNN) - Long Short-Term Memory (LSTM) model for regression. It demonstrates an end-to-end workflow from data loading to model evaluation, achieving a training Mean Absolute Error (MAE) of approximately 22 and a test MAE of 34.

The code is designed for educational and research purposes, showcasing machine learning techniques in prognostics and health management (PHM). It handles multi-subset data (FD001–FD004), computes RUL labels, and trains a deep learning model to forecast engine degradation.

## Overview

The script performs the following high-level steps:
1. **Data Loading and Merging**: Reads and combines training, testing, and RUL files from the CMAPSS dataset.
2. **Data Cleaning**: Removes unnecessary columns and handles NaN values.
3. **RUL Calculation**: Computes cycle-based RUL for both training (full lifecycles) and testing (partial lifecycles with ground-truth offsets).
4. **Feature Engineering**: Creates unique engine identifiers and normalizes sensor features using Min-Max scaling.
5. **Sequence Generation**: Builds time-series windows (default: 10 cycles) for input to the model.
6. **Model Building and Training**: Constructs a CNN-LSTM hybrid model using TensorFlow/Keras and trains it on the prepared data.
7. **Evaluation**: Assesses performance with custom accuracy, MSE, and MAE on train/test sets.

The dataset includes 26 features per cycle (operational settings and sensor readings) across hundreds of simulated engines under varying conditions. The output is a regression prediction of RUL in cycles.

## Problem Statement

In aviation and industrial sectors, equipment failures can lead to costly downtime, safety risks, and operational inefficiencies. Aircraft engines, in particular, degrade over time due to factors like wear, environmental conditions, and usage patterns. The challenge is to predict the Remaining Useful Life (RUL) of an engine before failure, enabling proactive maintenance.

The NASA CMAPSS dataset simulates turbofan engine degradation, providing time-series sensor data from multiple engines until failure (in training) or partial operation (in testing). The problem is a regression task: given historical sensor readings, forecast how many cycles remain before the engine fails. Key challenges include:
- Handling variable-length sequences across engines.
- Dealing with noisy, high-dimensional sensor data.
- Accounting for different operating regimes (e.g., FD001–FD004 subsets vary in fault modes and conditions).
- Avoiding overfitting in time-series modeling.

Without accurate predictions, maintenance is either reactive (post-failure) or scheduled (inefficient), leading to unnecessary costs or risks.

## Solution Approach

To address the RUL prediction problem, this script adopts a data-driven approach using deep learning for time-series analysis:

1. **Data Preparation**:
   - Merge subsets into unified DataFrames for train/test/RUL.
   - Assign unique engine IDs and compute RUL: For training, RUL decreases linearly to 0 at failure; for testing, it's offset by ground-truth values.
   - Normalize features to [0,1] range to improve model convergence.

2. **Time-Series Sequencing**:
   - Generate sliding windows of 10 cycles per engine to capture temporal dependencies. Each window forms a 3D input (samples × timesteps × features).
   - Note: The current implementation processes the entire DataFrame sequentially; for engine-specific windows, filter by `engine_id` in the loop (potential enhancement).

3. **Model Architecture**:
   - **CNN Component**: Uses 1D convolution (64 filters, kernel=3) and max pooling to extract local patterns from sensor sequences, reducing dimensionality.
   - **LSTM Component**: A 64-unit LSTM layer processes the sequential features to model long-term dependencies in degradation.
   - **Dense Layers**: Fully connected layers (64 units + output) for final RUL regression.
   - Compiled with Adam optimizer, MSE loss, and MAE metric.

4. **Training and Evaluation**:
   - Train for 20 epochs with 30% validation split and batch size 64.
   - Custom evaluation includes a normalized accuracy metric (1 - average error / RUL range), alongside MSE/MAE.
   - Results indicate good generalization, though test MAE is higher, suggesting room for hyperparameter tuning or data augmentation.

This supervised learning approach leverages the dataset's simulated failures to train a model that generalizes to unseen partial sequences.

## Technologies Used

- **Python 3.11**: Core language for scripting and data manipulation.
- **Pandas & NumPy**: For efficient data loading, merging, cleaning, and array operations.
- **Scikit-Learn**: MinMaxScaler for feature normalization and LabelEncoder for categorical handling; metrics like MAE/MSE for evaluation.
- **TensorFlow/Keras**: Deep learning framework for building and training the CNN-LSTM model. Key layers: Conv1D, MaxPooling1D, LSTM, Dense.
- **OS & Sys**: For file handling and progress tracking during processing.
- **Environment Assumptions**: Runs on platforms like Kaggle (with GPU support for faster training). No external installations needed beyond standard libraries.

The model has ~42k parameters, making it lightweight yet effective for time-series regression.

## Use Cases

This code serves multiple practical and educational applications:
- **Predictive Maintenance in Aviation**: Airlines can integrate similar models into monitoring systems to schedule engine inspections, reducing unscheduled downtime and extending asset life.
- **Industrial IoT (IIoT)**: Adaptable to other machinery (e.g., turbines, pumps) where sensor data tracks degradation, optimizing maintenance in manufacturing or energy sectors.
- **Research and Benchmarking**: PHM researchers can use this as a baseline for experimenting with advanced architectures (e.g., adding attention mechanisms) or handling imbalanced degradation patterns.
- **Educational Tool**: Teaches end-to-end ML workflows, from data preprocessing to deep learning for time-series forecasting. Ideal for courses in AI, data science, or reliability engineering.
- **Prototyping**: Quick setup for testing on CMAPSS; extend to real-world datasets like PHM Society challenges.

Example: In a fleet management system, input real-time sensor streams to predict RUL, triggering alerts when below a threshold (e.g., 50 cycles).

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/nasa-cmapss-rul-prediction.git
   cd nasa-cmapss-rul-prediction
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the CMAPSS Dataset**:
   - Obtain the dataset from [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/CMAPSS-Dataset/ff5v-kuh6).
   - Extract the files and place them in the `data/` directory.

5. **Run the Script**:
   ```bash
   python main.py
   ```

6. **Optional**: Enable GPU Support (if available):
   - Ensure TensorFlow is configured to use your GPU. Follow the [TensorFlow GPU setup guide](https://www.tensorflow.org/install/gpu).

7. **Verify Installation**:
   - Run the unit tests or a sample script to confirm everything is working:
     ```bash
     python test_script.py
     ```

You're now ready to explore predictive maintenance with the CMAPSS dataset!

## Results

- **Training**: MAE ~21.7, MSE ~2567, Custom Accuracy **~0.96**
- **Testing**: MAE ~33.7, MSE ~3274, Custom Accuracy **~0.93**

## Limitations and Known Issues

- The model's performance may vary depending on the preprocessing steps and hyperparameter tuning.
- The dataset is limited to specific engine types and may not generalize to other machinery.
- GPU support is recommended for faster training, but it is not mandatory.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

Please ensure your code adheres to the project's coding standards and includes relevant tests.

## License

This project is licensed under the MIT License.