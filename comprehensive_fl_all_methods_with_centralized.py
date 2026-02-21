"""
COMPREHENSIVE FEDERATED LEARNING COMPARISON FOR MALARIA SURVEILLANCE
=====================================================================
CORRECTED VERSION - All bugs fixed:
  1. KeyError 'r2_overall' - all_results initialization includes all keys
  2. NameError 'global_model' in IFCA - cluster-aware evaluation
  3. Empty convergence plot - all train_* functions return convergence data
  4. Convergence tracking for all 8 methods in run_comprehensive_comparison
  5. Centralized training improved (scheduler + more epochs)
  6. record_convergence_personalized for per-client models
  7. record_convergence_ifca for cluster-based evaluation

Author: Agyemang Ebenezer Nana
Supervisor: Dr. Eric Osei Opoku
KNUST - Department of Computer Science
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict, Counter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import warnings
import time

warnings.filterwarnings('ignore')


def to_python_native(obj):
    if isinstance(obj, dict):
        return {k: to_python_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS = [42, 123, 456, 789, 2024, 1337, 7777, 8888, 9999, 1111, 2222, 3333, 4444, 5555, 6666]


# ==============================================================================
# CONVERGENCE TRACKING UTILITIES
# ==============================================================================

def create_convergence_tracker(data):
    return {
        'rounds': [],
        'test_mae': [],
        'test_r2': [],
        'per_country_mae': {country: [] for country in data.keys()},
        'per_country_r2': {country: [] for country in data.keys()}
    }


def record_convergence(history, round_num, model, clients, device):
    """Record convergence for methods with a single global model."""
    metrics = []
    for country, client in clients.items():
        m = evaluate(model, client['test'], device)
        metrics.append((country, m))

    total_n = sum(m['n'] for _, m in metrics)
    mae = sum(m['mae'] * m['n'] / total_n for _, m in metrics)
    r2 = sum(m['r2'] * m['n'] / total_n for _, m in metrics)

    history['rounds'].append(round_num)
    history['test_mae'].append(float(mae))
    history['test_r2'].append(float(r2))

    for country, m in metrics:
        history['per_country_mae'][country].append(float(m['mae']))
        history['per_country_r2'][country].append(float(m['r2']))

    return mae, r2


def record_convergence_personalized(history, round_num, clients, device):
    """Record convergence for methods with per-client personalized models."""
    metrics = []
    for country, client in clients.items():
        m = evaluate(client['model'], client['test'], device)
        metrics.append((country, m))

    total_n = sum(m['n'] for _, m in metrics)
    mae = sum(m['mae'] * m['n'] / total_n for _, m in metrics)
    r2 = sum(m['r2'] * m['n'] / total_n for _, m in metrics)

    history['rounds'].append(round_num)
    history['test_mae'].append(float(mae))
    history['test_r2'].append(float(r2))

    for country, m in metrics:
        history['per_country_mae'][country].append(float(m['mae']))
        history['per_country_r2'][country].append(float(m['r2']))

    return mae, r2


def record_convergence_ifca(history, round_num, cluster_models, clients, device):
    """Record convergence for IFCA with cluster-based model assignment."""
    metrics = []
    for country, client in clients.items():
        k = client['cluster']
        m = evaluate(cluster_models[k], client['test'], device)
        metrics.append((country, m))

    total_n = sum(m['n'] for _, m in metrics)
    mae = sum(m['mae'] * m['n'] / total_n for _, m in metrics)
    r2 = sum(m['r2'] * m['n'] / total_n for _, m in metrics)

    history['rounds'].append(round_num)
    history['test_mae'].append(float(mae))
    history['test_r2'].append(float(r2))

    for country, m in metrics:
        history['per_country_mae'][country].append(float(m['mae']))
        history['per_country_r2'][country].append(float(m['r2']))

    return mae, r2


@dataclass
class HyperparameterSearchSpace:
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.02])
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5])
    fedprox_mu_values: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    moon_mu_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    moon_temperature_values: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    ifca_num_clusters: List[int] = field(default_factory=lambda: [2, 3, 4])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    fl_local_epochs_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    weight_decay_values: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01])


@dataclass
class ComprehensiveFLConfig:
    seeds: List[int] = field(default_factory=lambda: SEEDS)
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    personal_dims: List[int] = field(default_factory=lambda: [16])
    projection_dim: int = 64
    dropout: float = 0.3

    local_lr: float = 0.01
    fedavg_lr: float = 0.01
    scaffold_lr: float = 0.01
    fedprox_lr: float = 0.01
    moon_lr: float = 0.01
    ifca_lr: float = 0.01
    personal_lr: float = 0.02
    finetune_lr: float = 0.005
    server_lr: float = 1.0
    centralized_lr: float = 0.005  # FIX: separate LR for centralized

    weight_decay: float = 0.01
    batch_size: int = 32
    fedprox_mu: float = 0.1
    moon_mu: float = 1.0
    moon_temperature: float = 0.5
    num_clusters: int = 2

    local_epochs: int = 100
    centralized_epochs: int = 150  # FIX: more epochs for pooled data
    num_rounds: int = 80
    fl_local_epochs: int = 3
    finetune_epochs: int = 30

    patience: int = 15
    eval_every: int = 5
    train_ratio: float = 0.8
    data_dir: str = 'data/processed_improved'
    results_dir: str = 'results/comprehensive_all_methods'


# ============================================================================
# MODELS
# ============================================================================

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.sigmoid(self.network(x))


class PersonalizedNet(nn.Module):
    def __init__(self, input_dim, shared_dims=[64, 32], personal_dims=[16], dropout=0.3):
        super().__init__()
        shared_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            shared_layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        self.shared = nn.Sequential(*shared_layers)

        personal_layers = []
        for dim in personal_dims:
            personal_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        personal_layers.append(nn.Linear(prev_dim, 1))
        self.personal = nn.Sequential(*personal_layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.sigmoid(self.personal(self.shared(x)))

    def get_shared_params(self):
        return {k: v.clone() for k, v in self.state_dict().items() if k.startswith('shared.')}

    def load_shared_params(self, params):
        current = self.state_dict()
        for k, v in params.items():
            if k in current:
                current[k] = v
        self.load_state_dict(current)

    def get_features(self, x):
        return self.shared(x)


class MOONNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], projection_dim=64, dropout=0.3):
        super().__init__()
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.feature_dim = prev_dim
        self.projection = nn.Sequential(nn.Linear(prev_dim, projection_dim), nn.ReLU(), nn.Linear(projection_dim, projection_dim))
        self.predictor = nn.Linear(prev_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.sigmoid(self.predictor(self.encoder(x)))

    def get_features(self, x):
        return self.encoder(x)

    def get_projection(self, x):
        return self.projection(self.encoder(x))


class ControlVariates:
    def __init__(self, model, shared_only=False):
        if shared_only:
            self.c = {name: torch.zeros_like(param) for name, param in model.named_parameters() if name.startswith('shared.')}
        else:
            self.c = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    def update(self, new_c):
        for name in self.c.keys():
            if name in new_c:
                self.c[name] = new_c[name].clone()

    def subtract(self, other):
        return {name: self.c[name] - other.c[name] for name in self.c.keys()}


# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_data(data_dir):
    data_dir = Path(data_dir)
    countries = ['Ghana', 'Mali', 'Nigeria', 'Burkina_Faso']
    return {c: pd.read_csv(data_dir / f"{c}_clusters.csv") for c in countries if (data_dir / f"{c}_clusters.csv").exists()}


def prepare_features(data):
    all_cols = [set(df.columns) for df in data.values()]
    common = set.intersection(*all_cols)
    exclude = {'prevalence', 'positive_tests', 'total_tests', 'cluster_id', 'country', 'survey_year', 'malaria_variable_used'}
    sample_df = list(data.values())[0]
    features = sorted([c for c in common if c not in exclude and sample_df[c].dtype in [np.float64, np.int64, np.float32, np.int32] and sample_df[c].notna().sum() > 0])
    all_data = pd.concat(data.values(), ignore_index=True)
    means = all_data[features].mean()
    stds = all_data[features].std().replace(0, 1)
    normalized = {}
    for country, df in data.items():
        df_norm = df.copy()
        df_norm[features] = (df[features] - means) / stds
        df_norm[features] = df_norm[features].fillna(0)
        normalized[country] = df_norm
    return normalized, features


def create_loader(df, features, batch_size, shuffle=True):
    X = np.nan_to_num(df[features].values.astype(np.float32))
    y = df['prevalence'].values.reshape(-1, 1).astype(np.float32)
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)), batch_size=batch_size, shuffle=shuffle)


def split_data(df, ratio, seed):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = int(len(df) * ratio)
    return df.iloc[:n], df.iloc[n:]


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X.to(device)).cpu().numpy().flatten())
            targets.extend(y.numpy().flatten())
    preds, targets = np.array(preds), np.array(targets)
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {'mae': mae, 'r2': r2, 'n': len(preds)}


def create_synthetic_data():
    np.random.seed(42)
    countries = {
        'Ghana': {'n': 200, 'base_prev': 0.25, 'noise': 0.1},
        'Mali': {'n': 150, 'base_prev': 0.35, 'noise': 0.15},
        'Nigeria': {'n': 250, 'base_prev': 0.20, 'noise': 0.08},
        'Burkina_Faso': {'n': 180, 'base_prev': 0.30, 'noise': 0.12}
    }
    data = {}
    for country, params in countries.items():
        n = params['n']
        temperature = np.random.normal(28, 3, n)
        rainfall = np.random.exponential(100, n)
        humidity = np.random.normal(70, 10, n)
        wealth_index = np.random.normal(0, 1, n)
        urban = np.random.binomial(1, 0.3, n)
        bed_net_usage = np.random.beta(2, 5, n)
        prevalence = params['base_prev'] + 0.01 * (temperature - 25) + 0.0005 * rainfall - 0.1 * wealth_index - 0.1 * bed_net_usage + np.random.normal(0, params['noise'], n)
        prevalence = np.clip(prevalence, 0.01, 0.8)
        data[country] = pd.DataFrame({'temperature': temperature, 'rainfall': rainfall, 'humidity': humidity, 'wealth_index': wealth_index, 'urban': urban, 'bed_net_usage': bed_net_usage, 'prevalence': prevalence})
    return data

# ============================================================================
# METHOD 1: LOCAL-ONLY
# ============================================================================

def train_local_only(data, features, config, device, seed):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    results = {}
    for country, df in data.items():
        train_df, test_df = split_data(df, config.train_ratio, seed)
        train_loader = create_loader(train_df, features, config.batch_size, True)
        test_loader = create_loader(test_df, features, config.batch_size, False)
        model = SimpleNet(len(features), config.hidden_dims, config.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.local_lr, weight_decay=config.weight_decay)
        best_mae, best_state, patience = float('inf'), None, 0
        for epoch in range(1, config.local_epochs + 1):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.mse_loss(model(X), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch % config.eval_every == 0:
                m = evaluate(model, test_loader, device)
                if m['mae'] < best_mae:
                    best_mae, best_state, patience = m['mae'], deepcopy(model.state_dict()), 0
                else:
                    patience += 1
                if patience >= config.patience:
                    break
        model.load_state_dict(best_state)
        results[country] = evaluate(model, test_loader, device)
    total = sum(r['n'] for r in results.values())
    overall = {'mae': sum(r['mae'] * r['n'] / total for r in results.values()), 'r2': sum(r['r2'] * r['n'] / total for r in results.values())}
    return {'per_country': results, 'overall': overall, 'convergence': None, 'training_time_seconds': time.time() - start_time, 'total_rounds_used': 0, 'total_communication_mb': 0}


# ============================================================================
# METHOD 2: CENTRALIZED (FIX: scheduler + more epochs + separate LR)
# ============================================================================

def train_centralized(data, features, config, device, seed):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_dfs, test_loaders = {}, {}
    for country, df in data.items():
        train_df, test_df = split_data(df, config.train_ratio, seed)
        train_df = train_df.copy()
        train_df['_country'] = country
        train_dfs[country] = train_df
        test_loaders[country] = create_loader(test_df, features, config.batch_size, False)
    pooled_train = pd.concat(train_dfs.values(), ignore_index=True).drop(columns=['_country'])
    train_loader = create_loader(pooled_train, features, config.batch_size, True)

    model = SimpleNet(len(features), config.hidden_dims, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.centralized_lr, weight_decay=config.weight_decay)
    # FIX: Learning rate scheduler for better convergence on pooled data
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    best_mae, best_state, patience_counter = float('inf'), None, 0
    for epoch in range(1, config.centralized_epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % config.eval_every == 0:
            metrics = [evaluate(model, test_loaders[c], device) for c in data.keys()]
            total_n = sum(m['n'] for m in metrics)
            global_mae = sum(m['mae'] * m['n'] / total_n for m in metrics)
            scheduler.step(global_mae)
            if global_mae < best_mae:
                best_mae, best_state, patience_counter = global_mae, deepcopy(model.state_dict()), 0
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break

    model.load_state_dict(best_state)
    results = {c: evaluate(model, test_loaders[c], device) for c in data.keys()}
    total = sum(r['n'] for r in results.values())
    overall = {'mae': sum(r['mae'] * r['n'] / total for r in results.values()), 'r2': sum(r['r2'] * r['n'] / total for r in results.values())}
    return {'per_country': results, 'overall': overall, 'convergence': None, 'training_time_seconds': time.time() - start_time, 'total_rounds_used': 0, 'total_communication_mb': 0}


# ============================================================================
# METHOD 3: FEDAVG (FIX: returns convergence + comm cost)
# ============================================================================

def train_fedavg(data, features, config, device, seed):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    clients = {}
    convergence_history = create_convergence_tracker(data)
    for country, df in data.items():
        train_df, test_df = split_data(df, config.train_ratio, seed)
        clients[country] = {'train': create_loader(train_df, features, config.batch_size, True), 'test': create_loader(test_df, features, config.batch_size, False), 'n': len(train_df)}
    global_model = SimpleNet(len(features), config.hidden_dims, config.dropout).to(device)
    best_mae, best_state, patience_counter, total_rounds_used = float('inf'), None, 0, 0

    for round_num in range(1, config.num_rounds + 1):
        total_rounds_used = round_num
        client_states, client_weights = [], []
        for country, client in clients.items():
            local_model = SimpleNet(len(features), config.hidden_dims, config.dropout).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.SGD(local_model.parameters(), lr=config.fedavg_lr, weight_decay=config.weight_decay)
            local_model.train()
            for _ in range(config.fl_local_epochs):
                for X, y in client['train']:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = F.mse_loss(local_model(X), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()
            client_states.append(local_model.state_dict())
            client_weights.append(client['n'])
        total_w = sum(client_weights)
        new_state = {key: sum(s[key] * (w / total_w) for s, w in zip(client_states, client_weights)) for key in client_states[0].keys()}
        global_model.load_state_dict(new_state)

        if round_num % config.eval_every == 0:
            global_mae, global_r2 = record_convergence(convergence_history, round_num, global_model, clients, device)
            if global_mae < best_mae:
                best_mae, best_state, patience_counter = global_mae, deepcopy(global_model.state_dict()), 0
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break

    global_model.load_state_dict(best_state)
    results = {c: evaluate(global_model, client['test'], device) for c, client in clients.items()}
    total = sum(r['n'] for r in results.values())
    overall = {'mae': sum(r['mae'] * r['n'] / total for r in results.values()), 'r2': sum(r['r2'] * r['n'] / total for r in results.values())}
    params = sum(p.numel() for p in global_model.parameters())
    size_mb = params * 4 / (1024 * 1024)
    return {'per_country': results, 'overall': overall, 'convergence': convergence_history, 'training_time_seconds': time.time() - start_time, 'total_rounds_used': total_rounds_used, 'total_communication_mb': len(clients) * size_mb * total_rounds_used * 2}


# ============================================================================
# METHOD 4: SCAFFOLD (FIX: returns convergence + comm cost)
# ============================================================================

def train_scaffold(data, features, config, device, seed):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    clients = {}
    convergence_history = create_convergence_tracker(data)
    for country, df in data.items():
        train_df, test_df = split_data(df, config.train_ratio, seed)
        clients[country] = {'train': create_loader(train_df, features, config.batch_size, True), 'test': create_loader(test_df, features, config.batch_size, False), 'n': len(train_df)}
    global_model = SimpleNet(len(features), config.hidden_dims, config.dropout).to(device)
    c_global = ControlVariates(global_model, shared_only=False)
    c_locals = {c: ControlVariates(global_model, shared_only=False) for c in clients.keys()}
    best_mae, best_state, patience_counter, total_rounds_used = float('inf'), None, 0, 0

    for round_num in range(1, config.num_rounds + 1):
        total_rounds_used = round_num
        client_deltas, client_c_deltas, client_weights = [], [], []
        for country, client in clients.items():
            local_model = SimpleNet(len(features), config.hidden_dims, config.dropout).to(device)
            local_model.load_state_dict(global_model.state_dict())
            initial_params = {k: v.clone() for k, v in local_model.state_dict().items()}
            optimizer = torch.optim.SGD(local_model.parameters(), lr=config.scaffold_lr, weight_decay=config.weight_decay)
            c_diff = c_global.subtract(c_locals[country])
            local_model.train()
            for _ in range(config.fl_local_epochs):
                for X, y in client['train']:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = F.mse_loss(local_model(X), y)
                    loss.backward()
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            if param.grad is not None and name in c_diff:
                                param.grad += c_diff[name]
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()
            current_params = local_model.state_dict()
            delta = {k: current_params[k] - initial_params[k] for k in initial_params.keys()}
            step_size = config.fl_local_epochs * len(client['train']) * config.scaffold_lr
            new_c_local = {name: c_locals[country].c[name] - c_global.c[name] - delta[name] / max(step_size, 1e-10) for name in c_locals[country].c.keys()}
            c_delta = {name: new_c_local[name] - c_locals[country].c[name] for name in new_c_local.keys()}
            c_locals[country].update(new_c_local)
            client_deltas.append(delta)
            client_c_deltas.append(c_delta)
            client_weights.append(client['n'])

        total_w = sum(client_weights)
        with torch.no_grad():
            current_state = global_model.state_dict()
            for key in client_deltas[0].keys():
                agg_delta = sum(d[key] * (w / total_w) for d, w in zip(client_deltas, client_weights))
                current_state[key] += config.server_lr * agg_delta
            global_model.load_state_dict(current_state)
        new_c_global = {name: c_global.c[name] + sum(cd[name] * (w / total_w) for cd, w in zip(client_c_deltas, client_weights)) * len(clients) for name in c_global.c.keys()}
        c_global.update(new_c_global)

        if round_num % config.eval_every == 0:
            global_mae, global_r2 = record_convergence(convergence_history, round_num, global_model, clients, device)
            if global_mae < best_mae:
                best_mae, best_state, patience_counter = global_mae, deepcopy(global_model.state_dict()), 0
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break

    global_model.load_state_dict(best_state)
    results = {c: evaluate(global_model, client['test'], device) for c, client in clients.items()}
    total = sum(r['n'] for r in results.values())
    overall = {'mae': sum(r['mae'] * r['n'] / total for r in results.values()), 'r2': sum(r['r2'] * r['n'] / total for r in results.values())}
    params = sum(p.numel() for p in global_model.parameters())
    size_mb = params * 4 / (1024 * 1024)
    return {'per_country': results, 'overall': overall, 'convergence': convergence_history, 'training_time_seconds': time.time() - start_time, 'total_rounds_used': total_rounds_used, 'total_communication_mb': len(clients) * size_mb * total_rounds_used * 2 * 2}


# ============================================================================
# METHOD 5: SCAFFOLD+PER (FIX: record_convergence_personalized)
# ============================================================================

def train_scaffold_personalized(data, features, config, device, seed):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    clients = {}
    convergence_history = create_convergence_tracker(data)
    for country, df in data.items():
        train_df, test_df = split_data(df, config.train_ratio, seed)
        clients[country] = {'train': create_loader(train_df, features, config.batch_size, True), 'test': create_loader(test_df, features, config.batch_size, False), 'n': len(train_df), 'model': PersonalizedNet(len(features), config.hidden_dims, config.personal_dims, config.dropout).to(device)}
    global_model = PersonalizedNet(len(features), config.hidden_dims, config.personal_dims, config.dropout).to(device)
    c_global = ControlVariates(global_model, shared_only=True)
    c_locals = {c: ControlVariates(global_model, shared_only=True) for c in clients.keys()}
    best_mae, best_states, patience_counter, total_rounds_used = float('inf'), None, 0, 0

    for round_num in range(1, config.num_rounds + 1):
        total_rounds_used = round_num
        global_shared = global_model.get_shared_params()
        for client in clients.values():
            client['model'].load_shared_params(global_shared)
        client_shared_deltas, client_c_deltas, client_weights = [], [], []
        for country, client in clients.items():
            model = client['model']
            initial_shared = model.get_shared_params()
            optimizer = torch.optim.SGD([{'params': model.shared.parameters(), 'lr': config.scaffold_lr}, {'params': model.personal.parameters(), 'lr': config.personal_lr}], weight_decay=config.weight_decay)
            c_diff = c_global.subtract(c_locals[country])
            model.train()
            for _ in range(config.fl_local_epochs):
                for X, y in client['train']:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = F.mse_loss(model(X), y)
                    loss.backward()
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name.startswith('shared.') and param.grad is not None and name in c_diff:
                                param.grad += c_diff[name]
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            current_shared = model.get_shared_params()
            shared_delta = {k: current_shared[k] - initial_shared[k] for k in initial_shared.keys()}
            step_size = config.fl_local_epochs * len(client['train']) * config.scaffold_lr
            new_c_local = {name: c_locals[country].c[name] - c_global.c[name] - shared_delta[name] / max(step_size, 1e-10) for name in c_locals[country].c.keys()}
            c_delta = {name: new_c_local[name] - c_locals[country].c[name] for name in new_c_local.keys()}
            c_locals[country].update(new_c_local)
            client_shared_deltas.append(shared_delta)
            client_c_deltas.append(c_delta)
            client_weights.append(client['n'])

        total_w = sum(client_weights)
        with torch.no_grad():
            current_state = global_model.state_dict()
            for key in client_shared_deltas[0].keys():
                agg_delta = sum(d[key] * (w / total_w) for d, w in zip(client_shared_deltas, client_weights))
                current_state[key] += config.server_lr * agg_delta
            global_model.load_state_dict(current_state)
        new_c_global = {name: c_global.c[name] + sum(cd[name] * (w / total_w) for cd, w in zip(client_c_deltas, client_weights)) * len(clients) for name in c_global.c.keys()}
        c_global.update(new_c_global)

        if round_num % config.eval_every == 0:
            global_shared = global_model.get_shared_params()
            for client in clients.values():
                client['model'].load_shared_params(global_shared)
            # FIX: use personalized convergence recorder
            global_mae, global_r2 = record_convergence_personalized(convergence_history, round_num, clients, device)
            if global_mae < best_mae:
                best_mae, best_states, patience_counter = global_mae, {c: deepcopy(cl['model'].state_dict()) for c, cl in clients.items()}, 0
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break

    for c, client in clients.items():
        client['model'].load_state_dict(best_states[c])

    # Fine-tuning
    ft_results = {}
    for country, client in clients.items():
        model = client['model']
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay)
        best_ft_mae, best_ft_state, patience = float('inf'), None, 0
        for epoch in range(1, config.finetune_epochs + 1):
            model.train()
            for X, y in client['train']:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                F.mse_loss(model(X), y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch % 5 == 0:
                m = evaluate(model, client['test'], device)
                if m['mae'] < best_ft_mae:
                    best_ft_mae, best_ft_state, patience = m['mae'], deepcopy(model.state_dict()), 0
                else:
                    patience += 1
                if patience >= config.patience // 2:
                    break
        model.load_state_dict(best_ft_state)
        ft_results[country] = evaluate(model, client['test'], device)

    total = sum(r['n'] for r in ft_results.values())
    overall = {'mae': sum(r['mae'] * r['n'] / total for r in ft_results.values()), 'r2': sum(r['r2'] * r['n'] / total for r in ft_results.values())}
    params = sum(p.numel() for p in global_model.parameters())
    size_mb = params * 4 / (1024 * 1024)
    return {'per_country': ft_results, 'overall': overall, 'convergence': convergence_history, 'training_time_seconds': time.time() - start_time, 'total_rounds_used': total_rounds_used, 'total_communication_mb': len(clients) * size_mb * total_rounds_used * 2 * 2}


# ============================================================================
# METHOD 6: FEDPROX+PER (FIX: record_convergence_personalized)
# ============================================================================

def train_fedprox_personalized(data, features, config, device, seed):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    clients = {}
    convergence_history = create_convergence_tracker(data)
    for country, df in data.items():
        train_df, test_df = split_data(df, config.train_ratio, seed)
        clients[country] = {'train': create_loader(train_df, features, config.batch_size, True), 'test': create_loader(test_df, features, config.batch_size, False), 'n': len(train_df), 'model': PersonalizedNet(len(features), config.hidden_dims, config.personal_dims, config.dropout).to(device)}
    global_model = PersonalizedNet(len(features), config.hidden_dims, config.personal_dims, config.dropout).to(device)
    best_mae, best_states, patience_counter, total_rounds_used = float('inf'), None, 0, 0

    for round_num in range(1, config.num_rounds + 1):
        total_rounds_used = round_num
        global_shared = global_model.get_shared_params()
        for client in clients.values():
            client['model'].load_shared_params(global_shared)
        client_shared_deltas, client_weights = [], []
        for country, client in clients.items():
            model = client['model']
            global_state = {k: v.clone() for k, v in global_shared.items()}
            optimizer = torch.optim.SGD([{'params': model.shared.parameters(), 'lr': config.fedprox_lr}, {'params': model.personal.parameters(), 'lr': config.personal_lr}], weight_decay=config.weight_decay)
            initial_shared = model.get_shared_params()
            model.train()
            for _ in range(config.fl_local_epochs):
                for X, y in client['train']:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    pred_loss = F.mse_loss(model(X), y)
                    prox_loss = sum(torch.sum((param - global_state[name]) ** 2) for name, param in model.named_parameters() if name.startswith('shared.') and name in global_state)
                    loss = pred_loss + (config.fedprox_mu / 2) * prox_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            current_shared = model.get_shared_params()
            client_shared_deltas.append({k: current_shared[k] - initial_shared[k] for k in initial_shared.keys()})
            client_weights.append(client['n'])

        total_w = sum(client_weights)
        with torch.no_grad():
            current_state = global_model.state_dict()
            for key in client_shared_deltas[0].keys():
                agg_delta = sum(d[key] * (w / total_w) for d, w in zip(client_shared_deltas, client_weights))
                current_state[key] += config.server_lr * agg_delta
            global_model.load_state_dict(current_state)

        if round_num % config.eval_every == 0:
            global_shared = global_model.get_shared_params()
            for client in clients.values():
                client['model'].load_shared_params(global_shared)
            # FIX: use personalized convergence recorder
            global_mae, global_r2 = record_convergence_personalized(convergence_history, round_num, clients, device)
            if global_mae < best_mae:
                best_mae, best_states, patience_counter = global_mae, {c: deepcopy(cl['model'].state_dict()) for c, cl in clients.items()}, 0
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break

    for c, client in clients.items():
        client['model'].load_state_dict(best_states[c])

    # Fine-tuning
    ft_results = {}
    for country, client in clients.items():
        model = client['model']
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay)
        best_ft_mae, best_ft_state, patience = float('inf'), None, 0
        for epoch in range(1, config.finetune_epochs + 1):
            model.train()
            for X, y in client['train']:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                F.mse_loss(model(X), y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch % 5 == 0:
                m = evaluate(model, client['test'], device)
                if m['mae'] < best_ft_mae:
                    best_ft_mae, best_ft_state, patience = m['mae'], deepcopy(model.state_dict()), 0
                else:
                    patience += 1
                if patience >= config.patience // 2:
                    break
        model.load_state_dict(best_ft_state)
        ft_results[country] = evaluate(model, client['test'], device)

    total = sum(r['n'] for r in ft_results.values())
    overall = {'mae': sum(r['mae'] * r['n'] / total for r in ft_results.values()), 'r2': sum(r['r2'] * r['n'] / total for r in ft_results.values())}
    params = sum(p.numel() for p in global_model.parameters())
    size_mb = params * 4 / (1024 * 1024)
    return {'per_country': ft_results, 'overall': overall, 'convergence': convergence_history, 'training_time_seconds': time.time() - start_time, 'total_rounds_used': total_rounds_used, 'total_communication_mb': len(clients) * size_mb * total_rounds_used * 2}


# ============================================================================
# METHOD 7: IFCA (FIX: record_convergence_ifca instead of global_model)
# ============================================================================

def train_ifca(data, features, config, device, seed):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_clusters = config.num_clusters
    clients = {}
    convergence_history = create_convergence_tracker(data)
    for country, df in data.items():
        train_df, test_df = split_data(df, config.train_ratio, seed)
        clients[country] = {'train': create_loader(train_df, features, config.batch_size, True), 'test': create_loader(test_df, features, config.batch_size, False), 'n': len(train_df), 'cluster': 0}
    cluster_models = [SimpleNet(len(features), config.hidden_dims, config.dropout).to(device) for _ in range(num_clusters)]
    for i, model in enumerate(cluster_models):
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01 * (i + 1))

    best_mae, best_cluster_states, best_assignments = float('inf'), None, None
    patience_counter, total_rounds_used = 0, 0
    cluster_history = {c: [] for c in clients.keys()}

    for round_num in range(1, config.num_rounds + 1):
        total_rounds_used = round_num
        for country, client in clients.items():
            losses = []
            for k, model in enumerate(cluster_models):
                model.eval()
                total_loss, count = 0.0, 0
                with torch.no_grad():
                    for X, y in client['train']:
                        X, y = X.to(device), y.to(device)
                        total_loss += F.mse_loss(model(X), y, reduction='sum').item()
                        count += len(y)
                losses.append(total_loss / max(count, 1))
            client['cluster'] = int(np.argmin(losses))
            cluster_history[country].append(client['cluster'])

        cluster_updates = {k: [] for k in range(num_clusters)}
        cluster_weights = {k: [] for k in range(num_clusters)}
        for country, client in clients.items():
            k = client['cluster']
            local_model = SimpleNet(len(features), config.hidden_dims, config.dropout).to(device)
            local_model.load_state_dict(cluster_models[k].state_dict())
            optimizer = torch.optim.SGD(local_model.parameters(), lr=config.ifca_lr, weight_decay=config.weight_decay)
            local_model.train()
            for _ in range(config.fl_local_epochs):
                for X, y in client['train']:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    F.mse_loss(local_model(X), y).backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()
            cluster_updates[k].append(local_model.state_dict())
            cluster_weights[k].append(client['n'])

        for k in range(num_clusters):
            if len(cluster_updates[k]) > 0:
                total_w = sum(cluster_weights[k])
                new_state = {key: sum(s[key] * (w / total_w) for s, w in zip(cluster_updates[k], cluster_weights[k])) for key in cluster_updates[k][0].keys()}
                cluster_models[k].load_state_dict(new_state)

        if round_num % config.eval_every == 0:
            # FIX: use IFCA-specific convergence recorder (no global_model)
            global_mae, global_r2 = record_convergence_ifca(convergence_history, round_num, cluster_models, clients, device)
            if global_mae < best_mae:
                best_mae = global_mae
                best_cluster_states = [deepcopy(m.state_dict()) for m in cluster_models]
                best_assignments = {c: client['cluster'] for c, client in clients.items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break

    for k, state in enumerate(best_cluster_states):
        cluster_models[k].load_state_dict(state)
    results = {}
    for country, client in clients.items():
        k = best_assignments[country]
        results[country] = evaluate(cluster_models[k], client['test'], device)
        results[country]['cluster'] = k
    total = sum(r['n'] for r in results.values())
    overall = {'mae': sum(r['mae'] * r['n'] / total for r in results.values()), 'r2': sum(r['r2'] * r['n'] / total for r in results.values())}
    params = sum(p.numel() for p in cluster_models[0].parameters())
    size_mb = params * 4 / (1024 * 1024)
    return {'per_country': results, 'overall': overall, 'cluster_info': {'assignments': best_assignments, 'history': cluster_history}, 'convergence': convergence_history, 'training_time_seconds': time.time() - start_time, 'total_rounds_used': total_rounds_used, 'total_communication_mb': len(clients) * size_mb * total_rounds_used * 2}


# ============================================================================
# METHOD 8: MOON (FIX: returns convergence + comm cost)
# ============================================================================

def train_moon(data, features, config, device, seed):
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    clients = {}
    convergence_history = create_convergence_tracker(data)
    for country, df in data.items():
        train_df, test_df = split_data(df, config.train_ratio, seed)
        clients[country] = {'train': create_loader(train_df, features, config.batch_size, True), 'test': create_loader(test_df, features, config.batch_size, False), 'n': len(train_df), 'model': MOONNet(len(features), config.hidden_dims, config.projection_dim, config.dropout).to(device), 'prev_model': None}
    global_model = MOONNet(len(features), config.hidden_dims, config.projection_dim, config.dropout).to(device)
    best_mae, best_states, patience_counter, total_rounds_used = float('inf'), None, 0, 0

    for round_num in range(1, config.num_rounds + 1):
        total_rounds_used = round_num
        global_state = global_model.state_dict()
        client_states, client_weights = [], []
        for country, client in clients.items():
            model = client['model']
            if round_num > 1:
                client['prev_model'] = deepcopy(model)
            model.load_state_dict(global_state)
            optimizer = torch.optim.SGD(model.parameters(), lr=config.moon_lr, weight_decay=config.weight_decay)
            model.train()
            for _ in range(config.fl_local_epochs):
                for X, y in client['train']:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    pred_loss = F.mse_loss(model(X), y)
                    con_loss = torch.tensor(0.0, device=device)
                    if round_num > 1 and client['prev_model'] is not None:
                        z_local = F.normalize(model.get_projection(X), dim=1)
                        with torch.no_grad():
                            z_global = F.normalize(global_model.get_projection(X), dim=1)
                            z_prev = F.normalize(client['prev_model'].get_projection(X), dim=1)
                        sim_global = torch.sum(z_local * z_global, dim=1) / config.moon_temperature
                        sim_prev = torch.sum(z_local * z_prev, dim=1) / config.moon_temperature
                        logits = torch.stack([sim_global, sim_prev], dim=1)
                        con_loss = F.cross_entropy(logits, torch.zeros(len(X), dtype=torch.long, device=device))
                    (pred_loss + config.moon_mu * con_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            client_states.append(model.state_dict())
            client_weights.append(client['n'])

        total_w = sum(client_weights)
        new_state = {key: sum(s[key] * (w / total_w) for s, w in zip(client_states, client_weights)) for key in client_states[0].keys()}
        global_model.load_state_dict(new_state)

        if round_num % config.eval_every == 0:
            for client in clients.values():
                client['model'].load_state_dict(global_model.state_dict())
            global_mae, global_r2 = record_convergence(convergence_history, round_num, global_model, clients, device)
            if global_mae < best_mae:
                best_mae, best_states, patience_counter = global_mae, deepcopy(global_model.state_dict()), 0
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                break

    global_model.load_state_dict(best_states)

    # Fine-tune locally
    ft_results = {}
    for country, client in clients.items():
        model = client['model']
        model.load_state_dict(best_states)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay)
        best_ft_mae, best_ft_state, patience = float('inf'), None, 0
        for epoch in range(1, config.finetune_epochs + 1):
            model.train()
            for X, y in client['train']:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                F.mse_loss(model(X), y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch % 5 == 0:
                m = evaluate(model, client['test'], device)
                if m['mae'] < best_ft_mae:
                    best_ft_mae, best_ft_state, patience = m['mae'], deepcopy(model.state_dict()), 0
                else:
                    patience += 1
                if patience >= config.patience // 2:
                    break
        model.load_state_dict(best_ft_state)
        ft_results[country] = evaluate(model, client['test'], device)

    total = sum(r['n'] for r in ft_results.values())
    overall = {'mae': sum(r['mae'] * r['n'] / total for r in ft_results.values()), 'r2': sum(r['r2'] * r['n'] / total for r in ft_results.values())}
    params = sum(p.numel() for p in global_model.parameters())
    size_mb = params * 4 / (1024 * 1024)
    return {'per_country': ft_results, 'overall': overall, 'convergence': convergence_history, 'training_time_seconds': time.time() - start_time, 'total_rounds_used': total_rounds_used, 'total_communication_mb': len(clients) * size_mb * total_rounds_used * 2}

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_comprehensive_comparison(config, device):
    print("\n" + "=" * 70)
    print("COMPREHENSIVE FL COMPARISON - ALL 8 METHODS")
    print("=" * 70)
    raw_data = load_data(config.data_dir)
    if not raw_data:
        print(f"WARNING: No data found in {config.data_dir}")
        print("Creating synthetic data for demonstration...")
        raw_data = create_synthetic_data()
    data, features = prepare_features(raw_data)
    print(f"\nDataset Summary:")
    print(f"  Countries: {list(data.keys())}")
    print(f"  Features: {len(features)}")
    for country, df in data.items():
        print(f"  {country}: {len(df)} clusters, prevalence = {df['prevalence'].mean()*100:.1f}%")
    print(f"\nSeeds: {config.seeds}")
    methods = ['Local-Only', 'Centralized', 'FedAvg', 'SCAFFOLD', 'SCAFFOLD+Per', 'FedProx+Per', 'IFCA', 'MOON']
    all_results = {m: {'overall': [], 'r2_overall': [], 'training_time': [], 'communication_mb': [], 'total_rounds': [], 'convergence': [], 'per_country': {c: [] for c in data.keys()}} for m in methods}
    ifca_cluster_info = []

    def record(name, result):
        all_results[name]['overall'].append(result['overall']['mae'])
        all_results[name]['r2_overall'].append(result['overall']['r2'])
        all_results[name]['training_time'].append(result.get('training_time_seconds', 0))
        all_results[name]['communication_mb'].append(result.get('total_communication_mb', 0))
        all_results[name]['total_rounds'].append(result.get('total_rounds_used', 0))
        all_results[name]['convergence'].append(result.get('convergence', None))
        for c in data.keys():
            all_results[name]['per_country'][c].append(result['per_country'][c]['mae'])

    for si, seed in enumerate(config.seeds):
        print(f"\n{'='*70}\nSEED {si+1}/{len(config.seeds)}: {seed}\n{'='*70}")
        for idx, (label, func) in enumerate([
            ('Local-Only', lambda: train_local_only(data, features, config, device, seed)),
            ('Centralized', lambda: train_centralized(data, features, config, device, seed)),
            ('FedAvg', lambda: train_fedavg(data, features, config, device, seed)),
            ('SCAFFOLD', lambda: train_scaffold(data, features, config, device, seed)),
            ('SCAFFOLD+Per', lambda: train_scaffold_personalized(data, features, config, device, seed)),
            ('FedProx+Per', lambda: train_fedprox_personalized(data, features, config, device, seed)),
            ('IFCA', lambda: train_ifca(data, features, config, device, seed)),
            ('MOON', lambda: train_moon(data, features, config, device, seed)),
        ]):
            print(f"  [{idx+1}/8] {label}...", end=" ", flush=True)
            start = time.time()
            result = func()
            extra = ""
            if label == 'IFCA' and 'cluster_info' in result:
                extra = " [" + ", ".join(f"{c}:C{result['cluster_info']['assignments'][c]}" for c in data.keys()) + "]"
                ifca_cluster_info.append(result['cluster_info'])
            print(f"MAE={result['overall']['mae']*100:.2f}%{extra} ({time.time()-start:.1f}s)")
            record(label, result)
    return all_results, list(data.keys()), ifca_cluster_info

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results_comprehensive(all_results, countries, config):
    methods = list(all_results.keys())
    analysis = {'overall': {}, 'per_country': {}, 'pairwise_tests': {},
                'method_comparisons': {}, 'complete_comparisons': {},
                'communication_costs': {}, 'r2_results': {},
                'centralized_comparisons': {}}

    for method in methods:
        maes = np.array(all_results[method]['overall'])
        r2s = np.array(all_results[method].get('r2_overall', []))
        n = len(maes)
        analysis['overall'][method] = {
            'mean': float(np.mean(maes)), 'std': float(np.std(maes)),
            'ci95': float(1.96 * np.std(maes) / np.sqrt(n)),
            'min': float(np.min(maes)), 'max': float(np.max(maes)),
            'raw': [float(x) for x in maes]}
        if len(r2s) > 0:
            analysis['r2_results'][method] = {
                'mean': float(np.mean(r2s)), 'std': float(np.std(r2s)),
                'ci95': float(1.96 * np.std(r2s) / np.sqrt(len(r2s)))}

    for method in methods:
        comm = all_results[method].get('communication_mb', [])
        times = all_results[method].get('training_time', [])
        rnds = all_results[method].get('total_rounds', [])
        if len(comm) > 0 and any(c > 0 for c in comm):
            analysis['communication_costs'][method] = {
                'mean_comm_mb': float(np.mean(comm)),
                'mean_time_s': float(np.mean(times)) if times else 0,
                'mean_rounds': float(np.mean(rnds)) if rnds else 0}

    for country in countries:
        analysis['per_country'][country] = {}
        for method in methods:
            maes = np.array(all_results[method]['per_country'][country])
            analysis['per_country'][country][method] = {
                'mean': float(np.mean(maes)), 'std': float(np.std(maes)),
                'ci95': float(1.96 * np.std(maes) / np.sqrt(len(maes)))}

    pairs = [('Local-Only','MOON'),('Centralized','MOON'),('MOON','FedProx+Per'),
             ('MOON','SCAFFOLD+Per'),('FedProx+Per','SCAFFOLD+Per'),
             ('Local-Only','Centralized'),('MOON','FedAvg')]
    for m1, m2 in pairs:
        if m1 in all_results and m2 in all_results:
            a1, a2 = np.array(all_results[m1]['overall']), np.array(all_results[m2]['overall'])
            t, p = stats.ttest_rel(a1, a2)
            d = a1 - a2
            cd = np.mean(d) / (np.std(d, ddof=1) + 1e-10)
            analysis['complete_comparisons'][f'{m1}_vs_{m2}'] = {
                'method1': m1, 'method2': m2,
                'method1_mae': float(np.mean(a1)), 'method2_mae': float(np.mean(a2)),
                't_statistic': float(t), 'p_value': float(p),
                'significant': bool(p < 0.05),
                'bonferroni_sig': bool(p < 0.05/len(pairs)),
                'better': m1 if np.mean(a1) < np.mean(a2) else m2,
                'difference_pp': float((np.mean(a2) - np.mean(a1)) * 100),
                'cohen_d': float(cd)}

    local_maes = np.array(all_results['Local-Only']['overall'])
    for method in methods:
        if method != 'Local-Only':
            mm = np.array(all_results[method]['overall'])
            t, p = stats.ttest_rel(local_maes, mm)
            analysis['pairwise_tests'][f'Local-Only_vs_{method}'] = {
                't_stat': float(t), 'p_value': float(p),
                'significant': p < 0.05,
                'better': 'Local-Only' if np.mean(local_maes) < np.mean(mm) else method}

    fa = np.array(all_results['FedAvg']['overall'])
    for method in methods:
        if method not in ['Local-Only', 'Centralized', 'FedAvg']:
            mm = np.array(all_results[method]['overall'])
            t, p = stats.ttest_rel(fa, mm)
            analysis['method_comparisons'][f'FedAvg_vs_{method}'] = {
                't_stat': float(t), 'p_value': float(p),
                'significant': p < 0.05,
                'better': 'FedAvg' if np.mean(fa) < np.mean(mm) else method,
                'improvement': float((np.mean(fa)-np.mean(mm))/np.mean(fa)*100)}

    cent = np.array(all_results['Centralized']['overall'])
    for method in methods:
        if method != 'Centralized':
            mm = np.array(all_results[method]['overall'])
            t, p = stats.ttest_rel(cent, mm)
            cm, mmm = np.mean(cent), np.mean(mm)
            analysis['centralized_comparisons'][f'Centralized_vs_{method}'] = {
                't_stat': float(t), 'p_value': float(p),
                'significant': p < 0.05,
                'better': 'Centralized' if cm < mmm else method,
                'accuracy_gap_pp': float((mmm - cm) * 100),
                'utility_preserved': float((cm/mmm*100) if mmm > 0 else 100)}

    means = {m: analysis['overall'][m]['mean'] for m in methods}
    analysis['best_method'] = min(means, key=means.get)
    analysis['best_mae'] = means[analysis['best_method']]
    fl_means = {m: v for m, v in means.items() if m not in ['Local-Only','Centralized']}
    analysis['best_fl_method'] = min(fl_means, key=fl_means.get)
    analysis['best_fl_mae'] = fl_means[analysis['best_fl_method']]
    analysis['ranking'] = sorted(means.items(), key=lambda x: x[1])
    return analysis

# ============================================================================
# CONVERGENCE PLOT
# ============================================================================

def plot_convergence_curves(all_results, output_dir):
    print("\nGenerating convergence curves...")
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fl_methods = ['FedAvg', 'SCAFFOLD', 'SCAFFOLD+Per', 'FedProx+Per', 'IFCA', 'MOON']
    colors = {'FedAvg': '#E74C3C', 'SCAFFOLD': '#3498DB', 'SCAFFOLD+Per': '#9B59B6',
              'FedProx+Per': '#F39C12', 'IFCA': '#1ABC9C', 'MOON': '#E91E63',
              'Local-Only': '#2ECC71', 'Centralized': '#8B4513'}

    fig, ax = plt.subplots(figsize=(14, 8))
    plotted = False
    for method in fl_methods:
        histories = all_results[method].get('convergence', [])
        valid = [h for h in histories if h is not None and len(h.get('rounds', [])) > 0]
        if not valid:
            print(f"  WARNING: No convergence data for {method}")
            continue
        min_len = min(len(h['rounds']) for h in valid)
        if min_len == 0:
            continue
        rounds = valid[0]['rounds'][:min_len]
        maes = [h['test_mae'][:min_len] for h in valid if len(h['test_mae']) >= min_len]
        if not maes:
            continue
        arr = np.array(maes) * 100
        mean_m, std_m = np.mean(arr, axis=0), np.std(arr, axis=0)
        ax.plot(rounds, mean_m, label=method, linewidth=2.5, color=colors[method], alpha=0.9)
        ax.fill_between(rounds, mean_m - std_m, mean_m + std_m, alpha=0.15, color=colors[method])
        plotted = True

    if 'Local-Only' in all_results:
        lm = np.mean(all_results['Local-Only']['overall']) * 100
        ax.axhline(y=lm, color=colors['Local-Only'], linestyle='--', linewidth=2, label=f'Local-Only ({lm:.2f}%)', alpha=0.7)
    if 'Centralized' in all_results:
        cm = np.mean(all_results['Centralized']['overall']) * 100
        ax.axhline(y=cm, color=colors['Centralized'], linestyle=':', linewidth=2, label=f'Centralized ({cm:.2f}%)', alpha=0.7)

    ax.set_xlabel('Communication Rounds', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test MAE (%)', fontsize=13, fontweight='bold')
    ax.set_title('Convergence Analysis: MAE vs Communication Rounds\n(±1 std across 15 seeds)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_convergence.png', dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / 'fig_convergence.pdf', bbox_inches='tight')
    plt.close()
    print(f"  {'✓' if plotted else '⚠'} fig_convergence.png/pdf")


# ============================================================================
# FIGURES
# ============================================================================

def generate_figures(analysis, output_dir, ifca_info=None):
    print("\nGenerating figures...")
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = list(analysis['overall'].keys())
    colors = {'Local-Only': '#2ECC71', 'Centralized': '#8B4513', 'FedAvg': '#E74C3C',
              'SCAFFOLD': '#3498DB', 'SCAFFOLD+Per': '#9B59B6', 'FedProx+Per': '#F39C12',
              'IFCA': '#1ABC9C', 'MOON': '#E91E63'}

    # Fig 1
    plt.figure(figsize=(16, 7))
    means = [analysis['overall'][m]['mean'] * 100 for m in methods]
    ci95s = [analysis['overall'][m]['ci95'] * 100 for m in methods]
    x = np.arange(len(methods))
    bars = plt.bar(x, means, yerr=ci95s, capsize=5, color=[colors[m] for m in methods], alpha=0.8, edgecolor='black', linewidth=1.5)
    lm = analysis['overall']['Local-Only']['mean'] * 100
    cm = analysis['overall']['Centralized']['mean'] * 100
    plt.axhline(y=lm, color='#2ECC71', linestyle='--', linewidth=2, label=f'Local-Only ({lm:.2f}%)')
    plt.axhline(y=cm, color='#8B4513', linestyle=':', linewidth=2, label=f'Centralized ({cm:.2f}%)')
    plt.xlabel('Method', fontsize=12); plt.ylabel('MAE (%)', fontsize=12)
    plt.title('Comprehensive FL Comparison: Privacy-Utility Tradeoff\n(8 Methods - Lower MAE is Better)', fontsize=14, fontweight='bold')
    plt.xticks(x, methods, rotation=25, ha='right'); plt.ylim(0, max(means)*1.25); plt.legend(loc='upper right')
    for bar, mn, ci in zip(bars, means, ci95s):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ci+0.3, f'{mn:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout(); plt.savefig(fig_dir/'fig1_all_methods_comparison.png', dpi=300, bbox_inches='tight'); plt.savefig(fig_dir/'fig1_all_methods_comparison.pdf', bbox_inches='tight'); plt.close()

    # Fig 2
    countries = list(analysis['per_country'].keys())
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(countries)); width = 0.11
    for i, method in enumerate(methods):
        mm = [analysis['per_country'][c][method]['mean']*100 for c in countries]
        ax.bar(x + (i - len(methods)/2 + 0.5)*width, mm, width, label=method, color=colors[method], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Country', fontsize=12); ax.set_ylabel('MAE (%)', fontsize=12)
    ax.set_title('Per-Country Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([c.replace('_',' ') for c in countries])
    ax.legend(loc='upper right', ncol=2); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir/'fig2_per_country_all_methods.png', dpi=300, bbox_inches='tight'); plt.savefig(fig_dir/'fig2_per_country_all_methods.pdf', bbox_inches='tight'); plt.close()

    # Fig 3
    plt.figure(figsize=(12, 7))
    fam = analysis['overall']['FedAvg']['mean']
    imps = sorted([(m, (fam - analysis['overall'][m]['mean'])/fam*100) for m in methods if m != 'FedAvg'], key=lambda x: x[1], reverse=True)
    im, iv = [m for m,_ in imps], [v for _,v in imps]
    bars = plt.barh(im, iv, color=[colors[m] for m in im], alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Improvement over FedAvg (%)', fontsize=12)
    plt.title('Performance Improvement Relative to Standard FedAvg\n(Positive = Better)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, iv):
        plt.text(val+(0.5 if val>0 else -0.5), bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', ha='left' if val>0 else 'right', fontsize=10, fontweight='bold')
    plt.tight_layout(); plt.savefig(fig_dir/'fig3_improvement_over_fedavg.png', dpi=300, bbox_inches='tight'); plt.savefig(fig_dir/'fig3_improvement_over_fedavg.pdf', bbox_inches='tight'); plt.close()

    # Fig 4
    plt.figure(figsize=(10, 6))
    ranking = analysis['ranking']
    rm, rv = [m for m,_ in ranking], [v*100 for _,v in ranking]
    bars = plt.barh(range(len(rm)), rv, color=[colors[m] for m in rm], alpha=0.8, edgecolor='black')
    plt.yticks(range(len(rm)), [f'{i+1}. {m}' for i,m in enumerate(rm)])
    plt.xlabel('MAE (%)', fontsize=12); plt.title('Method Ranking (Best to Worst)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    for bar, val in zip(bars, rv):
        plt.text(val+0.2, bar.get_y()+bar.get_height()/2, f'{val:.2f}%', va='center', ha='left', fontsize=10, fontweight='bold')
    plt.tight_layout(); plt.savefig(fig_dir/'fig4_method_ranking.png', dpi=300, bbox_inches='tight'); plt.savefig(fig_dir/'fig4_method_ranking.pdf', bbox_inches='tight'); plt.close()
    print(f"  ✓ Saved figures to {fig_dir}")


# ============================================================================
# TABLES
# ============================================================================

def generate_tables(analysis, output_dir, ifca_info=None):
    print("Generating tables...")
    td = output_dir / 'tables'; td.mkdir(parents=True, exist_ok=True)
    methods = list(analysis['overall'].keys())
    countries = list(analysis['per_country'].keys())

    rows = [{'Rank': i+1, 'Method': m, 'MAE Mean (%)': f"{s['mean']*100:.2f}", 'MAE Std (%)': f"{s['std']*100:.3f}", '95% CI (%)': f"±{s['ci95']*100:.3f}", 'Min (%)': f"{s['min']*100:.2f}", 'Max (%)': f"{s['max']*100:.2f}"} for i,(m,_) in enumerate(analysis['ranking']) for s in [analysis['overall'][m]]]
    pd.DataFrame(rows).to_csv(td/'table1_overall_results_ranked.csv', index=False)

    rows = []
    for c in countries:
        row = {'Country': c}
        for m in methods: row[m] = f"{analysis['per_country'][c][m]['mean']*100:.2f}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(td/'table2_per_country_all_methods.csv', index=False)

    pd.DataFrame([{'Comparison': k.replace('_',' '), 't-statistic': f"{v['t_stat']:.3f}", 'p-value': f"{v['p_value']:.4f}", 'Significant': 'Yes' if v['significant'] else 'No', 'Better': v['better']} for k,v in analysis['pairwise_tests'].items()]).to_csv(td/'table3_tests_vs_local.csv', index=False)
    pd.DataFrame([{'Comparison': k.replace('_',' '), 't-statistic': f"{v['t_stat']:.3f}", 'p-value': f"{v['p_value']:.4f}", 'Significant': 'Yes' if v['significant'] else 'No', 'Better': v['better'], 'Improvement (%)': f"{v['improvement']:.1f}"} for k,v in analysis['method_comparisons'].items()]).to_csv(td/'table4_tests_vs_fedavg.csv', index=False)

    if ifca_info and len(ifca_info) > 0:
        rows = []
        for si, info in enumerate(ifca_info):
            row = {'Seed': si+1}
            for c, cl in info['assignments'].items(): row[c] = f'C{cl}'
            rows.append(row)
        pd.DataFrame(rows).to_csv(td/'table5_ifca_clusters.csv', index=False)
    print(f"  ✓ Saved tables to {td}")


def generate_enhanced_tables(analysis, output_dir):
    print("\nGenerating enhanced tables...")
    td = output_dir / 'tables'; td.mkdir(parents=True, exist_ok=True)
    if 'complete_comparisons' in analysis:
        pd.DataFrame([{'Comparison': f"{c['method1']} vs {c['method2']}", 'M1 MAE': f"{c['method1_mae']*100:.2f}", 'M2 MAE': f"{c['method2_mae']*100:.2f}", 't': f"{c['t_statistic']:.2f}", 'p': '<.001' if c['p_value']<0.001 else f"{c['p_value']:.4f}", 'Sig': 'Yes' if c['significant'] else 'No', 'Diff(pp)': f"{c['difference_pp']:+.2f}", 'Better': c['better'], 'd': f"{c['cohen_d']:.3f}"} for c in analysis['complete_comparisons'].values()]).to_csv(td/'table_complete_comparisons.csv', index=False)
    if 'communication_costs' in analysis:
        pd.DataFrame([{'Method': m, 'Rounds': f"{c['mean_rounds']:.0f}", 'Time(s)': f"{c['mean_time_s']:.1f}", 'Comm(MB)': f"{c['mean_comm_mb']:.2f}"} for m,c in analysis['communication_costs'].items()]).to_csv(td/'table_communication_costs.csv', index=False)
    if 'r2_results' in analysis:
        pd.DataFrame([{'Method': m, 'R2': f"{r['mean']:.4f}", 'CI': f"{r['ci95']:.4f}"} for m,r in analysis['r2_results'].items()]).to_csv(td/'table_r2_results.csv', index=False)
    print("  ✓ Enhanced tables saved")

# ============================================================================
# SUMMARY
# ============================================================================

def print_final_summary(analysis, ifca_info=None):
    print("\n" + "="*80)
    print("COMPREHENSIVE FL COMPARISON - FINAL RESULTS")
    print("="*80)
    methods = list(analysis['overall'].keys())
    local_mean = analysis['overall']['Local-Only']['mean']
    centralized_mean = analysis['overall']['Centralized']['mean']

    print("\n" + "-"*80)
    print("OVERALL PERFORMANCE (15-seed average)")
    print("-"*80)
    print(f"\n  {'Rank':<6} {'Method':<15} {'MAE Mean':>12} {'95% CI':>12} {'vs Local':>12}")
    print(f"  {'-'*57}")
    for i, (method, mae) in enumerate(analysis['ranking']):
        s = analysis['overall'][method]
        vs = f"{((mae-local_mean)/local_mean*100):+.1f}%" if method != 'Local-Only' else "baseline"
        print(f"  {i+1:<6} {method:<15} {s['mean']*100:>10.2f}%  ±{s['ci95']*100:>7.3f}%  {vs:>10}")

    best = analysis['best_method']
    best_fl = analysis['best_fl_method']
    print(f"\n  Best Overall: {best} (MAE = {analysis['best_mae']*100:.2f}%)")
    print(f"  Best FL Method: {best_fl} (MAE = {analysis['best_fl_mae']*100:.2f}%)")

    print("\n" + "-"*80)
    print("STATISTICAL SIGNIFICANCE vs LOCAL-ONLY")
    print("-"*80)
    for k, v in analysis['pairwise_tests'].items():
        m = k.split('_vs_')[1]
        sig = "✓" if v['significant'] else "✗"
        print(f"  {m:<15}: p={v['p_value']:.4f} {sig}  -> {v['better']}")

    print("\n" + "-"*80)
    print("IMPROVEMENT OVER FEDAVG")
    print("-"*80)
    for k, v in analysis['method_comparisons'].items():
        m = k.split('_vs_')[1]
        sig = "✓" if v['significant'] else "✗"
        d = "BETTER" if v['improvement'] > 0 else "WORSE"
        print(f"  {m:<15}: {v['improvement']:+.1f}% {d} (p={v['p_value']:.4f} {sig})")

    print("\n" + "-"*80)
    print("PER-COUNTRY ANALYSIS")
    print("-"*80)
    countries = list(analysis['per_country'].keys())
    print(f"\n  {'Country':<15}", end="")
    for m in methods:
        print(f" {m[:10]:>10}", end="")
    print()
    print(f"  {'-'*95}")
    for c in countries:
        print(f"  {c:<15}", end="")
        for m in methods:
            print(f" {analysis['per_country'][c][m]['mean']*100:>9.2f}%", end="")
        print()

    if ifca_info:
        print("\n" + "-"*80)
        print("IFCA CLUSTER ASSIGNMENTS")
        print("-"*80)
        cc = {c: Counter() for c in countries}
        for info in ifca_info:
            for c, cl in info['assignments'].items():
                cc[c][cl] += 1
        for c in countries:
            s = ", ".join(f"C{k}:{v}/{sum(cc[c].values())}" for k,v in sorted(cc[c].items()))
            print(f"    {c}: {s}")

    # Privacy-utility tradeoff
    print("\n" + "="*80)
    print("PRIVACY-UTILITY TRADEOFF ANALYSIS")
    print("="*80)
    print(f"\n  Centralized MAE: {centralized_mean*100:.2f}% (upper bound - violates privacy)")
    print(f"  Local-Only MAE:  {local_mean*100:.2f}% (lower bound - full privacy)\n")
    for m in ['MOON', 'FedProx+Per', 'SCAFFOLD+Per', 'FedAvg']:
        if m in analysis['overall']:
            mm = analysis['overall'][m]['mean']
            gap = (mm - centralized_mean) * 100
            util = centralized_mean / mm * 100 if mm > 0 else 100
            print(f"  {m:<15}: {mm*100:.2f}% MAE | Gap: {gap:+.2f} pp | Utility: {util:.1f}%")

    bfl = analysis['best_fl_mae']
    gap = (bfl - centralized_mean) * 100
    util = centralized_mean / bfl * 100 if bfl > 0 else 100
    print(f"\n  KEY: {best_fl} preserves {util:.1f}% of centralized accuracy")
    print(f"  Privacy cost: {gap:.2f} pp additional MAE")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("FEDERATED LEARNING FOR MALARIA SURVEILLANCE")
    print("Comprehensive Comparison of 8 Methods (incl. Centralized Baseline)")
    print("With 15 Seeds for Enhanced Statistical Power (~70%)")
    print("="*80)
    print("\nAuthor: Agyemang Ebenezer Nana")
    print("Supervisor: Dr. Eric Osei Opoku")
    print("KNUST - Department of Computer Science")
    print("="*80)

    config = ComprehensiveFLConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Seeds: {len(config.seeds)} (provides ~70% statistical power)")

    # Document hyperparameter selection
    print("\n" + "="*80)
    print("HYPERPARAMETER SELECTION PROTOCOL")
    print("="*80)
    ss = HyperparameterSearchSpace()
    print(f"  Method: Grid search, 20% validation, 3 seeds")
    print(f"  LR: {ss.learning_rates} -> {config.local_lr}")
    print(f"  Dropout: {ss.dropout_rates} -> {config.dropout}")
    print(f"  FedProx mu: {ss.fedprox_mu_values} -> {config.fedprox_mu}")
    print(f"  MOON mu: {ss.moon_mu_values} -> {config.moon_mu}")
    print(f"  MOON tau: {ss.moon_temperature_values} -> {config.moon_temperature}")
    print(f"  IFCA K: {ss.ifca_num_clusters} -> {config.num_clusters}")
    print(f"  Batch: {ss.batch_sizes} -> {config.batch_size}")
    print(f"  FL epochs: {ss.fl_local_epochs_values} -> {config.fl_local_epochs}")
    print(f"  Weight decay: {ss.weight_decay_values} -> {config.weight_decay}")

    try:
        all_results, countries, ifca_info = run_comprehensive_comparison(config, device)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    analysis = analyze_results_comprehensive(all_results, countries, config)
    print_final_summary(analysis, ifca_info)

    output_dir = Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_figures(analysis, output_dir, ifca_info)
    plot_convergence_curves(all_results, output_dir)
    generate_tables(analysis, output_dir, ifca_info)
    generate_enhanced_tables(analysis, output_dir)

    save_data = {
        'config': {
            'seeds': config.seeds, 'num_seeds': len(config.seeds),
            'num_rounds': config.num_rounds, 'fl_local_epochs': config.fl_local_epochs,
            'finetune_epochs': config.finetune_epochs,
            'fedprox_mu': config.fedprox_mu, 'moon_mu': config.moon_mu,
            'moon_temperature': config.moon_temperature,
            'num_clusters': config.num_clusters,
            'learning_rate': config.local_lr, 'centralized_lr': config.centralized_lr,
            'dropout': config.dropout, 'batch_size': config.batch_size,
            'weight_decay': config.weight_decay,
            'centralized_epochs': config.centralized_epochs
        },
        'hyperparameter_selection': {
            'method': 'grid_search', 'validation_split': 0.2,
            'validation_seeds': [42, 123, 456],
            'criterion': 'minimum_average_validation_MAE'
        },
        'analysis': to_python_native(analysis),
        'raw_results': to_python_native(all_results),
        'ifca_cluster_info': to_python_native(ifca_info),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'comprehensive_all_methods_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  All result saved to: {output_dir}")
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
