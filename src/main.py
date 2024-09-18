import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
import dask.dataframe as dd
import optuna
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
RANDOM_SEED = 42
N_SPLITS = 5

# Define paths
MODEL_CONFIG_PATH = 'model_config.json'
BEST_MODEL_PATH = 'best_model.pt'
FINAL_MODEL_PATH = 'final_model.pt'

class AdvancedFeatureEngineering(BaseEstimator, TransformerMixin):
    """Custom transformer for advanced feature engineering."""
    def __init__(self, numeric_cols):
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Feature engineering for rank and elo differences with adjusted weighting
        X['rank_difference'] = X['winner_rank'] - X['loser_rank']
        X['elo_difference'] = (X['winner_eloRating'] - X['loser_eloRating']) * 2  # Further emphasizing ELO ratings

        # Optional feature engineering based on available columns
        if 'winner_eloRatingDelta' in X.columns and 'loser_eloRatingDelta' in X.columns:
            X['elo_differenceDelta'] = (X['winner_eloRatingDelta'] - X['loser_eloRatingDelta']) * 2
            X['elo_differenceDelta_abs'] = np.abs(X['elo_differenceDelta'])

        # Convert dates to numeric days since the earliest date
        try:
            X['date'] = pd.to_datetime(X['date'], format='%Y%m%d', errors='coerce')
            min_date = X['date'].min()
            X['date'] = (X['date'] - min_date).dt.days
        except Exception as e:
            logging.warning(f"Error converting dates: {str(e)}. Setting 'date' to NaN.")
            X['date'] = np.nan

        # Convert numeric columns to float, handling errors
        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        return X

def load_data(first_number=2, last_number=15):
    """Load and combine player match data from multiple CSV files."""
    dfs = []
    core_columns = ['date', 'tournament', 'winner_name', 'winner_rank', 'winner_eloRating', 
                    'loser_name', 'loser_rank', 'loser_eloRating']
    optional_columns = ['level', 'bestOf', 'surface', 'indoor', 'speed', 'round', 
                        'winner_seed', 'winner_country_name', 'winner_country_id', 
                        'winner_eloRatingDelta', 'loser_seed', 'loser_country_name', 
                        'loser_country_id', 'loser_eloRatingDelta', 'score', 'outcome', 'loser_entry']
    all_columns = core_columns + optional_columns

    dtype_dict = {
        'loser_entry': 'object',
        'outcome': 'object',
        'winner_rank': 'float64',
        'loser_rank': 'float64',
        'winner_eloRating': 'float64',
        'loser_eloRating': 'float64',
        'bestOf': 'float64',
        'speed': 'float64',
        'winner_eloRatingDelta': 'float64',
        'loser_eloRatingDelta': 'float64',
        'indoor': 'float64',
        'winner_seed': 'float64',
        'loser_seed': 'float64'
    }

    for i in range(first_number, last_number + 1):
        file_path = f'PlayerMatches{i}.csv'
        try:
            df = dd.read_csv(file_path, low_memory=False, assume_missing=True, dtype=dtype_dict)
            missing_core_columns = [col for col in core_columns if col not in df.columns]
            if missing_core_columns:
                logging.warning(f"Missing core columns in {file_path}: {missing_core_columns}. Skipping this file.")
                continue
            available_columns = [col for col in all_columns if col in df.columns]
            df = df[available_columns].drop_duplicates().compute()
            dfs.append(df)
            logging.info(f"Loaded {file_path}")
        except FileNotFoundError:
            logging.warning(f"{file_path} not found. Skipping this file.")
        except Exception as e:
            logging.warning(f"An error occurred while loading {file_path}: {str(e)}. Skipping this file.")

    if not dfs:
        raise ValueError("No valid data found.")

    combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    if combined_df.empty:
        raise ValueError("The combined dataframe is empty.")

    return combined_df

def determine_column_types(df):
    """Determine numeric and categorical column types in the dataframe."""
    numeric_cols = ['winner_rank', 'loser_rank', 'winner_eloRating', 'loser_eloRating']
    potential_numeric_cols = ['bestOf', 'speed', 'winner_eloRatingDelta', 'loser_eloRatingDelta', 'indoor', 'winner_seed', 'loser_seed']

    for col in potential_numeric_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) or df[col].str.isnumeric().all():
                numeric_cols.append(col)

    categorical_cols = [col for col in df.columns if col not in numeric_cols and col != 'date']

    return numeric_cols, categorical_cols

def preprocess_data(df, numeric_cols, categorical_cols):
    """Preprocess the dataframe, including encoding categorical variables and handling missing values."""
    logging.info(f"Shape before preprocessing: {df.shape}")

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    feature_engineer = AdvancedFeatureEngineering(numeric_cols)
    df = feature_engineer.fit_transform(df)

    logging.info(f"Shape after feature engineering: {df.shape}")

    # Handle NaN values in numeric and categorical columns
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logging.warning(f"Column {col} has {nan_count} NaN values")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(-1)

    # Additional check for NaNs in rank_difference
    if df['rank_difference'].isna().any():
        logging.error("NaN values found in 'rank_difference' after preprocessing. Identifying rows with NaN values...")
        missing_rank_rows = df[df['rank_difference'].isna()]
        logging.info(f"Rows with missing 'rank_difference':\n{missing_rank_rows[['winner_rank', 'loser_rank']]}")
        df.dropna(subset=['rank_difference'], inplace=True)  # Drop rows with NaN in 'rank_difference'
        logging.info(f"Shape after dropping NaN rows in 'rank_difference': {df.shape}")

    return df, label_encoders

class JointEmbeddedModel(pl.LightningModule):
    """A PyTorch Lightning module for a neural network with categorical embeddings and numeric inputs."""
    def __init__(self, categorical_dims, numerical_dim, embedding_dim, hidden_dim, dropout_rate=0.3, learning_rate=1e-3):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embedding_dim) for dim in categorical_dims])
        self.fc1 = nn.Linear(len(categorical_dims) * embedding_dim + numerical_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_num], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x).squeeze()

    def training_step(self, batch, batch_idx):
        x_cat, x_num, y = batch
        y_hat = self(x_cat, x_num)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)  # Using AdamW optimizer
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

def create_dataloader(X, y, batch_size=64):
    """Create a DataLoader for training and evaluation."""
    x_cat, x_num = X
    x_cat = torch.tensor(x_cat, dtype=torch.long)
    x_num = torch.tensor(x_num, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_cat, x_num, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def ensemble_predictions(models, X):
    """Aggregate predictions from an ensemble of models."""
    preds = [model.predict(X) for model in models]
    return np.mean(preds, axis=0)

def save_model_config(config, path):
    """Save the model configuration to a JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f)

def load_model_config(path):
    """Load the model configuration from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def objective(trial):
    """Objective function for hyperparameter optimization with Optuna."""
    embedding_dim = trial.suggest_int('embedding_dim', 16, 128)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    model = JointEmbeddedModel(categorical_dims, numerical_dim, embedding_dim, hidden_dim, dropout_rate, learning_rate)
    dataloader = create_dataloader(X_train, y_train, batch_size=batch_size)

    trainer = pl.Trainer(
        max_epochs=20, 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        devices=1, 
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(model, dataloader)

    val_predictions = model(torch.tensor(X_val[0], dtype=torch.long), torch.tensor(X_val[1], dtype=torch.float32)).detach().cpu().numpy()
    if np.isnan(y_val).any() or np.isnan(val_predictions).any():
        raise ValueError("Validation targets or predictions contain NaN values.")

    val_loss = mean_squared_error(y_val, val_predictions)

    return val_loss

def analyze_winning_streaks(model, X, df_subset, eps=0.5, min_samples=5, threshold=0.5):
    """Analyze winning streaks using the trained model and clustering techniques."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x_cat, x_num = X

    with torch.no_grad():
        embedded = [emb(torch.tensor(x_cat[:, i], dtype=torch.long).to(device)) for i, emb in enumerate(model.embeddings)]
        embeddings = torch.cat(embedded, dim=1).cpu().numpy()
        outputs = model(torch.tensor(x_cat, dtype=torch.long).to(device), 
                        torch.tensor(x_num, dtype=torch.float32).to(device)).cpu().numpy()

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)

    df_subset['cluster'] = labels
    df_subset['predicted_rank_difference'] = outputs

    df_subset['easy_draw'] = (df_subset['rank_difference'] - df_subset['predicted_rank_difference']) > threshold
    df_subset['hard_draw'] = (df_subset['predicted_rank_difference'] - df_subset['rank_difference']) > threshold

    results = df_subset.groupby('winner_name').agg({
        'cluster': 'count',
        'easy_draw': 'sum',
        'hard_draw': 'sum'
    }).reset_index()

    results['easy_draw_ratio'] = results['easy_draw'] / results['cluster']
    results['hard_draw_ratio'] = results['hard_draw'] / results['cluster']

    results.sort_values('hard_draw_ratio', ascending=False, inplace=True)
    results.to_csv('winning_streak_analysis.csv', index=False)

    logging.info(f"Analysis results saved to winning_streak_analysis.csv")

    return results

if __name__ == "__main__":
    try:
        df = load_data()
        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        numeric_columns, categorical_columns = determine_column_types(df)
        logging.info(f"Numeric columns: {numeric_columns}")
        logging.info(f"Categorical columns: {categorical_columns}")

        df, label_encoders = preprocess_data(df, numeric_columns, categorical_columns)
        logging.info(f"Data preprocessed. Shape after preprocessing: {df.shape}")

        # Ensure all numeric columns are properly handled
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} contains non-numeric data after preprocessing")

        if df.shape[0] < N_SPLITS:
            raise ValueError(f"Not enough samples ({df.shape[0]}) for {N_SPLITS}-fold cross-validation.")

        X_cat = df[categorical_columns].values
        X_num = df[numeric_columns].values.astype(float)
        y = df['rank_difference'].values.astype(float)

        # Remove NaN values from y
        if np.isnan(y).any():
            raise ValueError("Target variable contains NaN values.")

        logging.info(f"Shape of X_cat: {X_cat.shape}")
        logging.info(f"Shape of X_num: {X_num.shape}")
        logging.info(f"Shape of y: {y.shape}")

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        scores = []

        for train_index, val_index in kf.split(X_cat):
            X_cat_train, X_cat_val = X_cat[train_index], X_cat[val_index]
            X_num_train, X_num_val = X_num[train_index], X_num[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Additional NaN checks for validation and training sets
            if np.isnan(X_cat_train).any() or np.isnan(X_num_train).any() or np.isnan(y_train).any():
                raise ValueError("Training data contains NaN values.")
            if np.isnan(X_cat_val).any() or np.isnan(X_num_val).any() or np.isnan(y_val).any():
                raise ValueError("Validation data contains NaN values.")

            X_train = (X_cat_train, X_num_train)
            X_val = (X_cat_val, X_num_val)

            categorical_dims = [len(label_encoders[col].classes_) for col in categorical_columns]
            numerical_dim = len(numeric_columns)

            try:
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=100)  # Further increased trials for finer parameter search

                best_params = study.best_params
                logging.info(f"Best Hyperparameters: {best_params}")

                # Save the model configuration
                model_config = {
                    'categorical_dims': categorical_dims,
                    'numerical_dim': numerical_dim,
                    'embedding_dim': best_params['embedding_dim'],
                    'hidden_dim': best_params['hidden_dim'],
                    'dropout_rate': best_params['dropout_rate'],
                    'learning_rate': best_params['learning_rate']
                }
                save_model_config(model_config, MODEL_CONFIG_PATH)

                model = JointEmbeddedModel(**model_config)
                dataloader = create_dataloader(X_train, y_train, batch_size=best_params['batch_size'])

                trainer = pl.Trainer(
                    max_epochs=100,  # Further increased max_epochs for deeper training
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    devices=1,
                    logger=False,
                    enable_checkpointing=False
                )
                trainer.fit(model, dataloader)

                val_predictions = model(torch.tensor(X_val[0], dtype=torch.long), torch.tensor(X_val[1], dtype=torch.float32)).detach().cpu().numpy()
                if np.isnan(val_predictions).any():
                    raise ValueError("Validation predictions contain NaN values.")

                val_loss = mean_squared_error(y_val, val_predictions)
                scores.append(val_loss)

                # Save the model state
                torch.save(model.state_dict(), BEST_MODEL_PATH)

            except Exception as e:
                logging.error(f"An error occurred during optimization: {str(e)}")
                logging.error("Exception details:", exc_info=True)

        logging.info(f"Cross-Validation MSE: {np.mean(scores):.4f}")

        # Train ensemble models and evaluate
        ensemble_models = [
            RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED),
            GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_SEED),
            LinearRegression()
        ]

        # Check for NaNs in ensemble training data
        if np.isnan(np.hstack((X_cat, X_num))).any() or np.isnan(y).any():
            raise ValueError("Ensemble training data contains NaN values.")

        ensemble_models = [model.fit(np.hstack((X_cat, X_num)), y) for model in ensemble_models]
        ensemble_preds = ensemble_predictions(ensemble_models, np.hstack((X_cat, X_num)))
        ensemble_mse = mean_squared_error(y, ensemble_preds)

        logging.info(f"Ensemble Test MSE: {ensemble_mse:.4f}")

        # Load the best model configuration and state for final analysis
        if os.path.exists(BEST_MODEL_PATH) and os.path.exists(MODEL_CONFIG_PATH):
            model_config = load_model_config(MODEL_CONFIG_PATH)
            model = JointEmbeddedModel(**model_config)
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            model.eval()

            test_predictions = model(torch.tensor(X_cat, dtype=torch.long), torch.tensor(X_num, dtype=torch.float32)).detach().cpu().numpy()
            if np.isnan(test_predictions).any():
                raise ValueError("Test predictions contain NaN values.")

            test_mse = mean_squared_error(y, test_predictions)
            logging.info(f"Final Test MSE: {test_mse}")

            winning_streak_analysis = analyze_winning_streaks(model, (X_cat, X_num), df)
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            logging.info("Script execution completed successfully.")
        else:
            logging.error("Best model or configuration not found. Ensure training is completed before running analysis.")

    except Exception as e:
        logging.error(f"An error occurred during script execution: {str(e)}")
        logging.error("Exception details:", exc_info=True)
