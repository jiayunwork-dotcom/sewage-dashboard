import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

from data_processing import KEY_PREDICTORS


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 horizon: int = 4, output_dim: int = 3, dropout: float = 0.1):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, horizon * output_dim)
        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.horizon, self.output_dim)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
               hidden_dim: int = 64, epochs: int = 100, batch_size: int = 32,
               lr: float = 0.001, verbose: bool = False) -> Tuple[nn.Module, Dict]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = X_train.shape[2]
    horizon = y_train.shape[1]
    output_dim = y_train.shape[2]

    model = LSTMPredictor(input_dim, hidden_dim=hidden_dim, horizon=horizon, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    y_te = torch.FloatTensor(y_test).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            test_pred = model(X_te)
            test_loss = criterion(test_pred, y_te).item()
            test_losses.append(test_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}')

    model.eval()
    with torch.no_grad():
        train_pred = model(X_t).cpu().numpy()
        test_pred = model(X_te).cpu().numpy()

    metrics = compute_metrics(y_test, test_pred)

    return model, {'train_pred': train_pred, 'test_pred': test_pred,
                   'train_losses': train_losses, 'test_losses': test_losses,
                   'metrics': metrics}


def predict_lstm(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    X_t = torch.FloatTensor(X).to(device)
    model.eval()
    with torch.no_grad():
        return model(X_t).cpu().numpy()


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                  n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1,
                  verbose: bool = False) -> Tuple[object, Dict]:
    horizon = y_train.shape[1]
    output_dim = y_train.shape[2]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)

    base_model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                              learning_rate=learning_rate, objective='reg:squarederror',
                              verbosity=1 if verbose else 0, random_state=42)
    wrapper = MultiOutputRegressor(base_model)
    wrapper.fit(X_train_flat, y_train_flat)

    train_pred_flat = wrapper.predict(X_train_flat)
    test_pred_flat = wrapper.predict(X_test_flat)

    train_pred = train_pred_flat.reshape(-1, horizon, output_dim)
    test_pred = test_pred_flat.reshape(-1, horizon, output_dim)

    metrics = compute_metrics(y_test, test_pred)

    return wrapper, {'train_pred': train_pred, 'test_pred': test_pred, 'metrics': metrics}


def predict_xgboost(model: object, X: np.ndarray) -> np.ndarray:
    horizon = 4
    output_dim = 3
    X_flat = X.reshape(X.shape[0], -1)
    pred_flat = model.predict(X_flat)
    return pred_flat.reshape(-1, horizon, output_dim)


def train_fusion(lstm_model: nn.Module, xgb_model: object,
                 lstm_test_pred: np.ndarray, xgb_test_pred: np.ndarray,
                 y_test: np.ndarray, lstm_weight: float = 0.5) -> Dict:
    xgb_weight = 1.0 - lstm_weight
    fusion_pred = lstm_weight * lstm_test_pred + xgb_weight * xgb_test_pred
    metrics = compute_metrics(y_test, fusion_pred)
    return {'test_pred': fusion_pred, 'metrics': metrics,
            'lstm_weight': lstm_weight, 'xgb_weight': xgb_weight}


def predict_fusion(lstm_model: nn.Module, xgb_model: object, X: np.ndarray,
                   lstm_weight: float = 0.5) -> np.ndarray:
    lstm_pred = predict_lstm(lstm_model, X)
    xgb_pred = predict_xgboost(xgb_model, X)
    return lstm_weight * lstm_pred + (1.0 - lstm_weight) * xgb_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    metrics = {}
    for i, name in enumerate(KEY_PREDICTORS):
        yt = y_true[:, :, i].flatten()
        yp = y_pred[:, :, i].flatten()
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        mae = np.mean(np.abs(yt - yp))
        mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100
        metrics[name] = {'RMSE': float(rmse), 'MAE': float(mae), 'MAPE': float(mape)}
    return metrics


def estimate_violation_probability(model_type: str, y_test: np.ndarray, y_pred: np.ndarray,
                                   current_pred: np.ndarray, standards: Dict) -> Dict:
    probs = {}
    for i, name in enumerate(KEY_PREDICTORS):
        std = standards.get(name)
        if std is None:
            probs[name] = [0.0] * current_pred.shape[1]
            continue

        violations = []
        for h in range(y_test.shape[1]):
            pred_h = y_pred[:, h, i]
            true_h = y_test[:, h, i]
            error = pred_h - true_h
            threshold = std
            cond_pred = current_pred[:, h, i]
            pred_violations = cond_pred > threshold
            if pred_violations.any():
                err_at_violation = error[pred_h > threshold]
                prob = float(np.mean(err_at_violation > 0)) if len(err_at_violation) > 0 else 0.5
            else:
                prob = float(np.mean(y_test[:, h, i] > std))
            violations.append(min(max(prob, 0.0), 1.0))
        probs[name] = violations
    return probs
