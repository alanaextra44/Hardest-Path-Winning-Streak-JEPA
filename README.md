# Tennis Match Win Streak Analysis

This project aims to analyze the most challenging path to a tennis historical winning-streak using machine learning and deep learning techniques. The model incorporates advanced feature engineering, data preprocessing, and a combination of neural networks and ensemble models to achieve accurate analysis of the level of competition.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Advanced feature engineering, including rank and Elo rating differences.
- Data preprocessing with error handling and missing value management.
- PyTorch Lightning framework for building and training neural networks.
- Hyperparameter optimization using Optuna.
- Ensemble methods for improved accuracy.
- Winning streak analysis using clustering techniques.

## Installation

To get started with this project, you need to have Python 3.x installed. You can then install the required packages using pip:
pip install -r requirements.txt

## Usage
Place your match data CSV files (named PlayerMatches2.csv to PlayerMatches15.csv) in the project directory.

Run the script to load the data, preprocess it, and train the models:

python main.py

The script will save the best model and configuration for later analysis.

## Data
The project uses historical player match data in CSV format. Each file should contain the following columns:

date

tournament

winner_name

winner_rank

winner_eloRating

loser_name

loser_rank

loser_eloRating

Optional columns for enhanced feature engineering can also be included.

## Model Architecture
The project utilizes a custom neural network with:

Categorical embeddings for player names and other categorical features.

Fully connected layers to process both embedded and numerical input features.

Dropout layers for regularization.

Hyperparameter Tuning

Hyperparameter optimization is performed using Optuna, allowing for fine-tuning of:

Embedding dimensions

Hidden layer sizes

Learning rates

Dropout rates

Batch sizes

## Results

The model's performance is evaluated using mean squared error (MSE) on the validation set. Ensemble models are also trained and compared for additional insights.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more information
