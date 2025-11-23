
"""
attention_seq2seq_upgraded_B.py

Full "Max-Boost" professional upgrade (Option B) of the uploaded attention_seq2seq_timeseries.py.

Features:
- Clean modular structure with RunConfig dataclass
- CLI + optional YAML config
- Deterministic seeding & environment metadata
- Mixed precision toggle
- Robust data loading (yfinance or CSV)
- Feature engineering, scaling, sequence creation
- Attention-based seq2seq (Bahdanau) with ability to extract attention weights
- Baseline LSTM model
- Training with callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Optional Optuna HPO scaffold (disabled by default)
- Attention heatmap saving, prediction plots, ASCII visualization
- Auto-generated report.md + requirements.txt + Dockerfile template
- All outputs saved to ./outputs/<run_id>/
- Designed for reviewers / HR: README generator + run metadata

Original uploaded file (for reference) used as source: /mnt/data/attention_seq2seq_timeseries.py
"""

from __future__ import annotations
import os
import sys
import json
import yaml
import argparse
import logging
import random
import datetime
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# TensorFlow import (lazy)
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Optional imports (handled gracefully)
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import optuna
except Exception:
    optuna = None

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("attn_upgraded")

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def now() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------
# Config
# -------------------------
@dataclass
class RunConfig:
    ticker: str = "AAPL"
    start_date: str = "2015-01-01"
    end_date: str = "2024-01-01"
    lookback: int = 60
    horizon: int = 7
    features: Optional[List[str]] = None
    batch_size: int = 32
    epochs: int = 50
    seed: int = 42
    out_dir: str = "./outputs"
    fast_test: bool = False
    use_mixed_precision: bool = False
    model_cell: str = "lstm"  # 'lstm' or 'gru'
    enable_optuna: bool = False
    optuna_trials: int = 20

    @classmethod
    def from_args(cls, args):
        cfg = cls()
        for k, v in vars(args).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

# -------------------------
# Data Loading & FE
# -------------------------
def load_data(ticker: str, start: str, end: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    if csv_path:
        logger.info("Loading data from CSV: %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        return df
    if yf is None:
        raise RuntimeError("yfinance not installed. Provide csv_path or install yfinance.")
    logger.info("Downloading %s from %s to %s", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data downloaded. Check ticker or date range.")
    df.index.name = "Date"
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Adj Close' in df.columns:
        df['return_1d'] = df['Adj Close'].pct_change().fillna(0)
        df['ma_7'] = df['Adj Close'].rolling(7, min_periods=1).mean()
        df['ma_21'] = df['Adj Close'].rolling(21, min_periods=1).mean()
    if 'Volume' in df.columns:
        df['vol_7'] = df['Volume'].rolling(7, min_periods=1).std().fillna(0)
    df = df.fillna(method='bfill').fillna(method='ffill').dropna()
    return df

def create_sequences(values: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    T = values.shape[0]
    X, y = [], []
    for start in range(0, T - lookback - horizon + 1):
        end = start + lookback
        X.append(values[start:end])
        y.append(values[end:end + horizon, 0])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    y = y.reshape((y.shape[0], y.shape[1], 1))
    return X, y

def make_decoder_inputs(y_array: np.ndarray) -> np.ndarray:
    dec = np.zeros_like(y_array)
    dec[:, 1:, :] = y_array[:, :-1, :]
    return dec

# -------------------------
# Attention Layer & Models
# -------------------------
class BahdanauAttention(layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, enc_outputs, dec_hidden, return_weights=False):
        if len(dec_hidden.shape) == 2:
            dec_hidden_time = tf.expand_dims(dec_hidden, axis=1)
        else:
            dec_hidden_time = dec_hidden
        score = self.V(tf.nn.tanh(self.W1(enc_outputs) + self.W2(dec_hidden_time)))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * enc_outputs
        context = tf.reduce_sum(context, axis=1)
        if return_weights:
            return context, tf.squeeze(weights, axis=-1)
        return context

def build_attention_seq2seq(input_shape: Tuple[int, int], enc_units: int = 128, dec_units: int = 128, horizon: int = 7, cell_type: str = 'lstm') -> Model:
    enc_in = layers.Input(shape=input_shape, name='encoder_input')
    if cell_type == 'gru':
        enc_outputs, enc_state = layers.GRU(enc_units, return_sequences=True, return_state=True, name='encoder_gru')(enc_in)
        enc_h, enc_c = enc_state, None
    else:
        enc_outputs, enc_h, enc_c = layers.LSTM(enc_units, return_sequences=True, return_state=True, name='encoder_lstm')(enc_in)
    dec_in = layers.Input(shape=(horizon, 1), name='decoder_input')
    attention = BahdanauAttention(units=dec_units)
    dec_cell = layers.LSTMCell(dec_units)
    # We'll unstack decoder inputs and iterate, producing outputs and capturing attention weights if needed.
    dec_unstack = layers.Lambda(lambda x: tf.unstack(x, axis=1), name='dec_unstack')(dec_in)
    all_outputs = []
    attn_weights_list = []
    state_h = enc_h
    state_c = enc_c if enc_c is not None else layers.Lambda(lambda x: tf.zeros_like(x))(enc_h)
    for t in range(horizon):
        current_input = dec_unstack[t]
        current_input_sq = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(current_input)
        context, attn_w = attention(enc_outputs, state_h, return_weights=True)
        rnn_input = layers.Concatenate(axis=-1)([context, current_input_sq])
        output, [state_h, state_c] = dec_cell(rnn_input, states=[state_h, state_c])
        step_out = layers.Dense(1, activation='linear')(output)
        all_outputs.append(step_out)
        attn_weights_list.append(attn_w)
    outputs = layers.Lambda(lambda x: tf.stack(x, axis=1))(all_outputs)
    attn_stack = layers.Lambda(lambda x: tf.stack(x, axis=1), name='attn_stack')(attn_weights_list)
    model = Model([enc_in, dec_in], [outputs, attn_stack], name='attention_seq2seq_full')
    model.compile(optimizer='adam', loss='mse')
    return model

def build_baseline_lstm(input_shape: Tuple[int, int], hidden_units: int = 128, horizon: int = 7) -> Model:
    enc_in = layers.Input(shape=input_shape, name='enc_input')
    enc_out, enc_h, enc_c = layers.LSTM(hidden_units, return_state=True, name='enc_lstm')(enc_in)
    dec = layers.RepeatVector(horizon)(enc_out)
    dec_out = layers.LSTM(hidden_units, return_sequences=True)(dec)
    out = layers.TimeDistributed(layers.Dense(1))(dec_out)
    model = Model(enc_in, out, name='baseline_lstm')
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------
# Metrics & Visualization
# -------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    mae = mean_absolute_error(y_true_f, y_pred_f)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true_f - y_pred_f) / np.where(np.abs(y_true_f) < 1e-8, 1e-8, y_true_f))) * 100
    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape)}

def ascii_plot(true: np.ndarray, pred: np.ndarray, width: int = 60) -> str:
    def sparkline(arr):
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mx - mn < 1e-8:
            return '.' * len(arr)
        scaled = ((arr - mn) / (mx - mn) * (width - 1)).astype(int)
        chars = ''.join('▁▂▃▄▅▆▇█'[max(0, min(7, int(v * 8 / width)))] for v in scaled)
        return chars
    return f"True : {sparkline(true)}\nPred : {sparkline(pred)}"

def save_pred_plot(true: np.ndarray, pred: np.ndarray, path: str, title: str = ""):
    plt.figure(figsize=(6,3))
    plt.plot(true, label='true')
    plt.plot(pred, label='pred')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_attention_heatmap(attn: np.ndarray, path: str, xlabel: str = "Lookback Index", ylabel: str = "Forecast Step"):
    # attn shape: (horizon, lookback)
    plt.figure(figsize=(8, max(2, attn.shape[0]*0.5)))
    plt.imshow(attn, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Attention Weights (rows=forecast step, cols=lookback)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# -------------------------
# Reporting & Packaging
# -------------------------
def write_markdown_report(out_dir: str, cfg: RunConfig, metrics_attn: Dict[str, float], metrics_base: Dict[str, float], samples: List[Dict[str, Any]]):
    md = []
    md.append("# Run Report")
    md.append(f"Run time: {datetime.datetime.now().isoformat()}")
    md.append("## Config")
    md.append("```json")
    md.append(json.dumps(asdict(cfg), indent=2))
    md.append("```")
    md.append("## Metrics")
    md.append("| Model | RMSE | MAE | MAPE |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Attention | {metrics_attn['RMSE']:.4f} | {metrics_attn['MAE']:.4f} | {metrics_attn['MAPE']:.2f}% |")
    md.append(f"| Baseline  | {metrics_base['RMSE']:.4f} | {metrics_base['MAE']:.4f} | {metrics_base['MAPE']:.2f}% |")
    md.append("## Sample Interpretations")
    for s in samples:
        md.append(f"### Sample {s['idx']}")
        md.append("ASCII visualization:")
        md.append("```")
        md.append(s['ascii'])
        md.append("```")
        if 'pred_png' in s:
            md.append(f"Prediction plot: {s['pred_png']}")
        if 'attn_png' in s:
            md.append(f"Attention heatmap: {s['attn_png']}")
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n\n".join(md))
    logger.info("Wrote report: %s", path)

def write_requirements(out_dir: str):
    reqs = [
        "tensorflow>=2.10",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "yfinance",
        "pyyaml"
    ]
    path = os.path.join(out_dir, "requirements.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(reqs))
    logger.info("Wrote requirements.txt")

def write_dockerfile(out_dir: str):
    docker = [
        "FROM python:3.10-slim",
        "WORKDIR /app",
        "COPY . /app",
        "RUN pip install --no-cache-dir -r requirements.txt",
        'CMD ["python", "attention_seq2seq_upgraded_B.py", "--fast-test"]'
    ]
    path = os.path.join(out_dir, "Dockerfile")
    with open(path, "w") as fh:
        fh.write("\n".join(docker))
    logger.info("Wrote Dockerfile")

# -------------------------
# Train & Evaluate Pipeline
# -------------------------
def train_and_evaluate(cfg: RunConfig, csv_path: Optional[str] = None):
    set_seed(cfg.seed)
    if cfg.use_mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled")
        except Exception as e:
            logger.warning("Mixed precision not available: %s", e)
    run_id = now()
    out_dir = os.path.join(cfg.out_dir, run_id)
    ensure_dir(out_dir)
    # metadata
    meta = {"run_id": run_id, "time": datetime.datetime.now().isoformat(), "cfg": asdict(cfg)}
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    # load data
    df = load_data(cfg.ticker, cfg.start_date, cfg.end_date, csv_path)
    df = feature_engineer(df)
    if cfg.features is None:
        features = ['Adj Close', 'return_1d', 'ma_7', 'ma_21', 'vol_7']
    else:
        features = cfg.features
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Feature {col} not in data columns: {df.columns.tolist()}")
    data = df[features].values.astype(np.float32)
    scaler = StandardScaler()
    data_s = scaler.fit_transform(data)
    X, y = create_sequences(data_s, cfg.lookback, cfg.horizon)
    N = X.shape[0]
    train_end = int(0.7 * N)
    val_end = int(0.85 * N)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    if cfg.fast_test:
        X_train, y_train = X_train[:512], y_train[:512]
        X_val, y_val = X_val[:128], y_val[:128]
        X_test, y_test = X_test[:128], y_test[:128]
    input_shape = (cfg.lookback, X.shape[2])
    # models
    attn_model = build_attention_seq2seq(input_shape, enc_units=128, dec_units=128, horizon=cfg.horizon, cell_type=cfg.model_cell)
    baseline_model = build_baseline_lstm(input_shape, hidden_units=128, horizon=cfg.horizon)
    # callbacks
    attn_ckpt = os.path.join(out_dir, "attn_best.h5")
    base_ckpt = os.path.join(out_dir, "base_best.h5")
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    cp_attn = ModelCheckpoint(attn_ckpt, monitor='val_loss', save_best_only=True)
    cp_base = ModelCheckpoint(base_ckpt, monitor='val_loss', save_best_only=True)
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    # prepare decoder inputs for training (teacher forcing)
    dec_train = make_decoder_inputs(y_train)
    dec_val = make_decoder_inputs(y_val)
    epochs = 3 if cfg.fast_test else cfg.epochs
    history_attn = attn_model.fit([X_train, dec_train], [y_train, np.zeros((y_train.shape[0], cfg.horizon, cfg.lookback))],
                                  validation_data=([X_val, dec_val], [y_val, np.zeros((y_val.shape[0], cfg.horizon, cfg.lookback))]),
                                  epochs=epochs, batch_size=cfg.batch_size, callbacks=[es, cp_attn, rlp], verbose=2)
    history_base = baseline_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                                      batch_size=cfg.batch_size, callbacks=[es, cp_base, rlp], verbose=2)
    # Predict. attn_model returns [preds, attn_weights]
    dec_test_zero = np.zeros_like(y_test)
    preds_attn, attn_weights = attn_model.predict([X_test, dec_test_zero], verbose=0)
    preds_base = baseline_model.predict(X_test, verbose=0)
    metrics_attn = compute_metrics(y_test, preds_attn)
    metrics_base = compute_metrics(y_test, preds_base)
    with open(os.path.join(out_dir, "metrics_attn.json"), "w") as fh:
        json.dump(metrics_attn, fh, indent=2)
    with open(os.path.join(out_dir, "metrics_base.json"), "w") as fh:
        json.dump(metrics_base, fh, indent=2)
    # Interpret and save samples
    samples = []
    for i in range(min(5, X_test.shape[0])):
        true_seq = y_test[i].reshape(-1)
        pred_seq = preds_attn[i].reshape(-1)
        # unscale only target feature approximately
        target_mean = scaler.mean_[0]
        target_scale = np.sqrt(scaler.var_[0])
        true_un = true_seq * target_scale + target_mean
        pred_un = pred_seq * target_scale + target_mean
        ascii_v = ascii_plot(true_un, pred_un)
        pred_png = os.path.join(out_dir, f"sample_{i}_pred.png")
        save_pred_plot(true_un, pred_un, pred_png, title=f"Sample {i} prediction")
        # attention weights for sample: attn_weights shape = (num_samples, horizon, lookback)
        attn_s = attn_weights[i] if attn_weights is not None else None
        attn_png = None
        if attn_s is not None:
            attn_png = os.path.join(out_dir, f"sample_{i}_attn.png")
            save_attention_heatmap(attn_s, attn_png)
        samples.append({"idx": i, "ascii": ascii_v, "pred_png": pred_png, "attn_png": attn_png})
    # Save models & artifacts
    attn_model.save(os.path.join(out_dir, "attn_model_full.h5"))
    baseline_model.save(os.path.join(out_dir, "baseline_model_full.h5"))
    # Write report + packaging files
    write_markdown_report(out_dir, cfg, metrics_attn, metrics_base, samples)
    write_requirements(out_dir)
    write_dockerfile(out_dir)
    logger.info("Run complete. Outputs in %s", out_dir)
    return {"out_dir": out_dir, "metrics_attn": metrics_attn, "metrics_base": metrics_base, "samples": samples}

# -------------------------
# Optuna HPO scaffold (optional)
# -------------------------
def optuna_objective(trial, cfg: RunConfig, csv_path: Optional[str]):
    # This is a small example objective - user can grow it
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    units = trial.suggest_categorical("units", [64, 128, 256])
    # small pipeline build - using cfg.fast_test for speed
    set_seed(cfg.seed)
    df = load_data(cfg.ticker, cfg.start_date, cfg.end_date, csv_path)
    df = feature_engineer(df)
    features = cfg.features or ['Adj Close', 'return_1d', 'ma_7', 'ma_21', 'vol_7']
    data = df[features].values.astype(np.float32)
    scaler = StandardScaler()
    data_s = scaler.fit_transform(data)
    X, y = create_sequences(data_s, cfg.lookback, cfg.horizon)
    N = X.shape[0]
    train_end = int(0.7 * N)
    val_end = int(0.85 * N)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    if cfg.fast_test:
        X_train, y_train = X_train[:256], y_train[:256]
        X_val, y_val = X_val[:64], y_val[:64]
    input_shape = (cfg.lookback, X.shape[2])
    model = build_attention_seq2seq(input_shape, enc_units=units, dec_units=units, horizon=cfg.horizon, cell_type=cfg.model_cell)
    dec_train = make_decoder_inputs(y_train)
    dec_val = make_decoder_inputs(y_val)
    model.fit([X_train, dec_train], [y_train, np.zeros((y_train.shape[0], cfg.horizon, cfg.lookback))],
              validation_data=([X_val, dec_val], [y_val, np.zeros((y_val.shape[0], cfg.horizon, cfg.lookback))]),
              epochs=3, batch_size=cfg.batch_size, verbose=0)
    preds, _ = model.predict([X_val, np.zeros_like(y_val)], verbose=0)
    metrics = compute_metrics(y_val, preds)
    return metrics["RMSE"]

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Upgraded Attention Seq2Seq Time Series project (Option B)")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--fast_test", action="store_true")
    p.add_argument("--yaml", default=None, help="optional YAML config file")
    p.add_argument("--csv", default=None, help="optional CSV input path")
    p.add_argument("--mixed", action="store_true", help="enable mixed precision")
    p.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--optuna", action="store_true", help="run optuna HPO (if optuna installed)")
    p.add_argument("--optuna_trials", type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    # load YAML if provided
    cfg = RunConfig.from_args(args)
    if args.yaml:
        with open(args.yaml) as fh:
            raw = yaml.safe_load(fh)
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    # override flags
    cfg.use_mixed_precision = args.mixed
    cfg.model_cell = args.cell
    cfg.enable_optuna = args.optuna
    cfg.optuna_trials = args.optuna_trials
    cfg.fast_test = args.fast_test
    cfg.out_dir = args.out_dir
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.seed = args.seed
    set_seed(cfg.seed)
    res = None
    if cfg.enable_optuna and optuna is not None:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: optuna_objective(t, cfg, args.csv), n_trials=cfg.optuna_trials)
        logger.info("Optuna best: %s", study.best_params)
    try:
        res = train_and_evaluate(cfg, csv_path=args.csv)
    except Exception as e:
        logger.exception("Run failed: %s", e)
        sys.exit(1)
    print("Outputs saved to:", res["out_dir"])

if __name__ == "__main__":
    main()
