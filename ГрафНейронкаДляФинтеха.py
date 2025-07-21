import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from collections import deque
import random
import time
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import random as py_random
import kagglehub
import pickle
import hashlib
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def check_gpu():
    global pynvml_available
    print("Проверка доступности GPU...")
    if not torch.cuda.is_available():
        print("CUDA недоступен. Обучение будет на CPU.")
        return torch.device("cpu"), None, None
    
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    cuda_version = torch.version.cuda
    total_memory = free_memory = None
    
    print(f"Устройство CUDA: {device}")
    print(f"Количество GPU: {gpu_count}")
    print(f"Текущий GPU: {gpu_name} (ID: {current_device})")
    print(f"Версия CUDA: {cuda_version}")
    
    if pynvml_available:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total / (1024 ** 3)
            free_memory = memory_info.free / (1024 ** 3)
            print(f"Общая память GPU: {total_memory:.2f} GB")
            print(f"Свободная память GPU: {free_memory:.2f} GB")
        except Exception as e:
            print(f"Предупреждение: Не удалось инициализировать pynvml: {e}")
    else:
        print("Информация о памяти GPU недоступна (pynvml отсутствует)")
    
    print(f"PyTorch будет использовать устройство: {device}")
    return device, gpu_name, total_memory

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    py_random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TSSGC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_transaction_types, output_dim, device):
        super(TSSGC, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.temporal_fc = nn.Linear(input_dim + 1, hidden_dim)
        self.semantic_emb = nn.Embedding(num_transaction_types, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device
        self.to(self.device)
    
    def forward(self, x, edge_index, edge_attr, transaction_types, timestamps):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else None
        transaction_types = transaction_types.to(self.device)
        timestamps = timestamps.to(self.device)
        
        spatial = self.conv1(x, edge_index)
        spatial = F.relu(spatial)
        spatial = self.batch_norm1(spatial)
        spatial = self.conv2(spatial, edge_index)
        spatial = F.relu(spatial)
        spatial = self.batch_norm2(spatial)
        spatial = self.conv3(spatial, edge_index)
        spatial = F.relu(spatial)
        spatial = self.batch_norm3(spatial)
        
        time_diffs = self.compute_time_diffs(timestamps, edge_index, edge_attr)
        temporal_input = torch.cat([x, time_diffs], dim=1)
        temporal = self.temporal_fc(temporal_input)
        temporal = F.relu(temporal)
        
        semantic = self.semantic_emb(transaction_types)
        
        spatial_w = F.softmax(self.attention(spatial), dim=0)
        temporal_w = F.softmax(self.attention(temporal), dim=0)
        semantic_w = F.softmax(self.attention(semantic), dim=0)
        combined = spatial_w * spatial + temporal_w * temporal + semantic_w * semantic
        
        out = self.fc(combined)
        return out
    
    def compute_time_diffs(self, timestamps, edge_index, edge_attr):
        time_diffs = torch.zeros((timestamps.size(0), 1), device=self.device)
        if edge_attr is not None:
            edge_attr = torch.clamp(edge_attr, min=-1e3, max=1e3)
            edge_attr = (edge_attr - edge_attr.mean()) / (edge_attr.std() + 1e-9)
            for i, (src, dst) in enumerate(edge_index.t()):
                time_diffs[src] += edge_attr[i]
                time_diffs[dst] += edge_attr[i]
        degree = torch.bincount(edge_index[0], minlength=timestamps.size(0)) + torch.bincount(edge_index[1], minlength=timestamps.size(0))
        degree = degree.clamp(min=1).unsqueeze(1).float()
        return time_diffs / degree

class DQN(nn.Module):
    def __init__(self, input_dim, action_space_size, device):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_size)
        self.device = device
        self.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class FraudGNNRL:
    def __init__(self, input_dim, hidden_dim, output_dim, num_transaction_types, action_space_size, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tssgc = TSSGC(input_dim, hidden_dim, num_transaction_types, output_dim, self.device).to(self.device)
        self.dqn = DQN(input_dim + output_dim + 4, action_space_size, self.device).to(self.device)
        self.target_dqn = DQN(input_dim + output_dim + 4, action_space_size, self.device).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer_tssgc = torch.optim.Adam(self.tssgc.parameters(), lr=config["tssgc_lr"])
        self.optimizer_dqn = torch.optim.Adam(self.dqn.parameters(), lr=config["dqn_lr"])
        self.replay_buffer = deque(maxlen=config["replay_buffer_size"])
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.batch_size = config["batch_size"]
        self.action_space_size = action_space_size
        self.thresholds = np.linspace(config["threshold_min"], config["threshold_max"], config["num_thresholds"])
        self.input_dim = input_dim
    
    def act(self, state):
        if not isinstance(state, np.ndarray) or state.shape[0] != self.input_dim + 4:
            raise ValueError(f"Неверная форма состояния: ожидается ({self.input_dim + 4},), получено {state.shape}")
        if random.random() < self.epsilon:
            return random.randrange(self.action_space_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_values = self.dqn(state)
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_dqn(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer_dqn.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        self.optimizer_dqn.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()
    
    def train_tssgc(self, data, labels, class_weights):
        self.optimizer_tssgc.zero_grad()
        edge_attr = data.edge_attr.to(self.device) if data.edge_attr is not None else None
        transaction_types = data.transaction_types.to(self.device)
        timestamps = data.timestamps.to(self.device)
    
        out = self.tssgc(data.x.to(self.device), 
                         data.edge_index.to(self.device),
                         edge_attr,
                         transaction_types,
                         timestamps)
    
        loss = F.binary_cross_entropy_with_logits(out.view(-1), 
                                                  labels.to(self.device).view(-1),
                                                  weight=class_weights.to(self.device))
        loss = torch.clamp(loss, max=100.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tssgc.parameters(), max_norm=1.0)
        self.optimizer_tssgc.step()
        return loss.item()
    
    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
    
    def predict(self, data, action):
        with torch.no_grad():
            if action >= self.action_space_size:
                raise ValueError(f"Действие {action} превышает размер пространства действий {self.action_space_size}")
            threshold_idx = action
            threshold = self.thresholds[threshold_idx]
            
            transaction_types = data.transaction_types.to(self.device)
            timestamps = data.timestamps.to(self.device)
            
            out = self.tssgc(data.x.to(self.device), data.edge_index.to(self.device),
                             data.edge_attr.to(self.device) if data.edge_attr is not None else None,
                             transaction_types,
                             timestamps)
            probs = torch.sigmoid(out).view(-1)
            return probs, (probs > threshold).float()

class FederatedServer:
    def __init__(self, tssgc_model, dqn_model, num_clients):
        self.global_tssgc = tssgc_model
        self.global_dqn = dqn_model
        self.num_clients = num_clients
        self.device = tssgc_model.device
    
    def aggregate(self, client_tssgcs, client_dqns, client_sizes):
        tssgc_dict = self.global_tssgc.state_dict()
        weights = torch.tensor([size / sum(client_sizes) for size in client_sizes], dtype=torch.float32).to(self.device)
        for k in tssgc_dict.keys():
            tssgc_dict[k] = torch.stack([client_tssgcs[i].state_dict()[k].to(self.device).float() for i in range(self.num_clients)], dim=0)
            tssgc_dict[k] = (tssgc_dict[k].transpose(0, -1) @ weights).transpose(0, -1)
        self.global_tssgc.load_state_dict(tssgc_dict)
        
        dqn_dict = self.global_dqn.state_dict()
        for k in dqn_dict.keys():
            dqn_dict[k] = torch.stack([client_dqns[i].state_dict()[k].to(self.device).float() for i in range(self.num_clients)], dim=0)
            dqn_dict[k] = (dqn_dict[k].transpose(0, -1) @ weights).transpose(0, -1)
        self.global_dqn.load_state_dict(dqn_dict)

class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
    
    def __call__(self, val_f1, model):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

def recall_at_k(y_true, y_scores, k_percent):
    k = max(1, int(len(y_true) * k_percent / 100))
    sorted_indices = np.argsort(y_scores)[::-1]
    top_k_indices = sorted_indices[:k]
    return np.sum(y_true[top_k_indices]) / (np.sum(y_true) + 1e-9)

def evaluate_adversarial_robustness(model, data, action, noise_level):
    with torch.no_grad():
        noisy_x = data.x.to(model.device) + torch.randn_like(data.x.to(model.device)) * noise_level
        noisy_data = Data(x=noisy_x, edge_index=data.edge_index.to(model.device), edge_attr=data.edge_attr.to(model.device) if data.edge_attr is not None else None,
                          transaction_types=data.transaction_types.to(model.device),
                          timestamps=data.timestamps.to(model.device), y=data.y.to(model.device))
        probs, predictions = model.predict(noisy_data, action)
        f1 = f1_score(data.y.cpu().numpy(), predictions.cpu().numpy(), zero_division=0)
        auc_roc = roc_auc_score(data.y.cpu().numpy(), probs.cpu().numpy())
        auc_pr = average_precision_score(data.y.cpu().numpy(), probs.cpu().numpy())
        recall_k = recall_at_k(data.y.cpu().numpy(), probs.cpu().numpy(), k_percent=1)
    return {"f1": f1, "auc_roc": auc_roc, "auc_pr": auc_pr, "recall_at_k": recall_k}

def filter_and_reindex_edges(edge_index, edge_attr, indices):
    try:
        indices = torch.tensor(indices, dtype=torch.long)
        mask = torch.isin(edge_index, indices)
        mask = mask[0] & mask[1]
        filtered_edges = edge_index[:, mask]
        filtered_attr = edge_attr[mask] if edge_attr is not None else None
        index_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
        reindexed_edges = torch.tensor([[index_map.get(int(edge), -1) for edge in filtered_edges[0]],
                                       [index_map.get(int(edge), -1) for edge in filtered_edges[1]]], dtype=torch.long)
        valid_mask = (reindexed_edges[0] >= 0) & (reindexed_edges[1] >= 0)
        filtered_edges = reindexed_edges[:, valid_mask]
        filtered_attr = filtered_attr[valid_mask] if filtered_attr is not None else None
        return filtered_edges, filtered_attr
    except Exception as e:
        print(f"Ошибка в filter_and_reindex_edges: {e}")
        return torch.tensor([[], []], dtype=torch.long), None

def download_kaggle_dataset(dataset_slug, cache_dir, local_path=None):
    start_time = time.time()
    print(f"Начало загрузки {dataset_slug}...")
    cache_file = os.path.join(cache_dir, f"{dataset_slug.replace('/', '_')}.csv")
    
    if local_path and os.path.exists(local_path):
        print(f"Использование локального файла {local_path} для {dataset_slug}")
        return local_path
    
    if os.path.exists(cache_file):
        print(f"Использование кэшированного датасета {dataset_slug} за {time.time() - start_time:.2f} секунд")
        return cache_file
    
    try:
        path = kagglehub.dataset_download(dataset_slug, timeout=15)
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError(f"CSV-файлы не найдены в {path}")
        file_path = files[0]
        os.makedirs(cache_dir, exist_ok=True)
        os.rename(file_path, cache_file)
        elapsed_time = time.time() - start_time
        print(f"Завершение загрузки {dataset_slug} за {elapsed_time:.2f} секунд")
        return cache_file
    except Exception as e:
        print(f"Ошибка загрузки {dataset_slug}: {e}")
        return None

def load_and_preprocess_data(dataset_name, file_path, config):
    start_time = time.time()
    print(f"Начало предобработки {dataset_name}...")
    
    params = {
        "dataset_name": dataset_name,
        "max_rows": config["max_rows"],
        "n_neighbors": config["n_neighbors"],
        "top_features": config["top_features"]
    }
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    cache_file = os.path.join(config["cache_dir"], f"{dataset_name}_processed_{params_hash}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data, input_dim = pickle.load(f)
            print(f"Загружены кэшированные данные для {dataset_name} за {time.time() - start_time:.2f} секунд")
            return data, input_dim
        except Exception as e:
            print(f"Ошибка загрузки кэш-файла {cache_file}: {e}")
            os.remove(cache_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stages = ["Чтение CSV", "Заполнение пропусков", "Кодирование категориальных", "Нормализация признаков", "Применение SMOTE", "Построение k-NN графа", "Сохранение кэша"]
    total_stages = len(stages)
    
    try:
        print(f"Этап предобработки 1/{total_stages}: {stages[0]}")
        stage_start = time.time()
        if dataset_name == "financial_fraud":
            label_col = "isFraud"
            time_col = "step"
            categorical_cols = ["type"]
            usecols = ["isFraud", "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        elif dataset_name == "credit_card_2023":
            label_col = "Class"
            time_col = None
            categorical_cols = []
            usecols = ["Class", "Amount"] + [f"V{i}" for i in range(1, 29)]
        
        df = pd.read_csv(file_path, usecols=[col for col in usecols if col in pd.read_csv(file_path, nrows=1).columns], nrows=config["max_rows"])
        print(f"Размер датасета: {len(df)} строк, {len(df.columns)} столбцов")
        print(f"Этап 1/{total_stages} завершен за {time.time() - stage_start:.2f} секунд")
        
        print(f"Этап предобработки 2/{total_stages}: {stages[1]}")
        stage_start = time.time()
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        print(f"Этап 2/{total_stages} завершен за {time.time() - stage_start:.2f} секунд")
        
        print(f"Этап предобработки 3/{total_stages}: {stages[2]}")
        stage_start = time.time()
        for col in categorical_cols:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
        print(f"Этап 3/{total_stages} завершен за {time.time() - stage_start:.2f} секунд")
        
        print(f"Этап предобработки 4/{total_stages}: {stages[3]}")
        stage_start = time.time()
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop(label_col, errors="ignore")
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        if time_col not in df.columns:
            df["TransactionDT"] = np.arange(len(df)) * config["time_step"]
            time_col = "TransactionDT"
        
        variances = df[numeric_cols].var()
        if config["top_features"] is None:
            numeric_cols = list(numeric_cols)
        else:
            top_features = variances.sort_values(ascending=False).index[:config["top_features"]]
            numeric_cols = list(top_features)
        print(f"Выбрано {len(numeric_cols)} лучших признаков: {numeric_cols}")
        print(f"Этап 4/{total_stages} завершен за {time.time() - stage_start:.2f} секунд")
        
        print(f"Этап предобработки 5/{total_stages}: {stages[4]}")
        stage_start = time.time()
        X = df[numeric_cols].values
        y = df[label_col].values
        timestamps_orig = df[time_col].values if time_col in df.columns else np.arange(len(df)) * config["time_step"]
        transaction_types_orig = df[categorical_cols[0]].values if categorical_cols else np.zeros(len(df), dtype=int)
        
        smote = SMOTE(random_state=config["seed"])
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        knn.fit(X)
        _, indices = knn.kneighbors(X_resampled)
        timestamps_resampled = timestamps_orig[indices].flatten()
        transaction_types_resampled = transaction_types_orig[indices].flatten() if categorical_cols else np.zeros(len(X_resampled), dtype=int)
        
        print(f"После SMOTE: {len(X_resampled)} образцов, распределение классов: {np.bincount(y_resampled.astype(int))}")
        print(f"Этап 5/{total_stages} завершен за {time.time() - stage_start:.2f} секунд")
        
        print(f"Этап предобработки 6/{total_stages}: {stages[5]}")
        stage_start = time.time()
        X_sample = X_resampled.copy()
        y_sample = y_resampled.copy()
        
        print(f"Размер X_sample: {len(X_sample)}")
        print(f"Размер y_sample: {len(y_sample)}")
        print(f"Размер timestamps_resampled: {len(timestamps_resampled)}")
        print(f"Размер transaction_types_resampled: {len(transaction_types_resampled)}")
        
        print("Обучение k-NN модели...")
        knn_start = time.time()
        knn = NearestNeighbors(n_neighbors=config["n_neighbors"], metric="cosine", n_jobs=-1)
        knn.fit(X_sample)
        print(f"k-NN модель обучена за {time.time() - knn_start:.2f} секунд")
        
        print("Поиск ближайших соседей...")
        search_start = time.time()
        distances, indices = knn.kneighbors(X_sample)
        print(f"Соседи найдены за {time.time() - search_start:.2f} секунд")
        
        print("Построение графа...")
        edge_index = []
        edge_attr = []
        total_nodes = len(X_sample)
        for i in tqdm(range(total_nodes), desc="Генерация рёбер графа"):
            neighbors = indices[i]
            for j in neighbors:
                if i != j and j < len(X_sample):
                    edge_index.append((i, j))
                    time_diff = timestamps_resampled[j] - timestamps_resampled[i]
                    edge_attr.append(time_diff / (np.std(timestamps_resampled) + 1e-9))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(device)
        print(f"Граф построен: {edge_index.size(1)} ребер")
        print(f"Этап 6/{total_stages} завершен за {time.time() - stage_start:.2f} секунд")
        
        print(f"Этап предобработки 7/{total_stages}: {stages[6]}")
        stage_start = time.time()
        data = Data(
            x=torch.FloatTensor(X_sample).to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            transaction_types=torch.LongTensor(transaction_types_resampled).to(device),
            timestamps=torch.FloatTensor(timestamps_resampled).to(device),
            y=torch.FloatTensor(y_sample).to(device)
        )
        input_dim = len(numeric_cols)
        
        os.makedirs(config["cache_dir"], exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((data, input_dim), f)
        print(f"Этап 7/{total_stages} завершен за {time.time() - stage_start:.2f} секунд")
        
        elapsed_time = time.time() - start_time
        print(f"Предобработка {dataset_name} завершена за {elapsed_time:.2f} секунд")
        return data, input_dim
    except Exception as e:
        print(f"Ошибка предобработки {dataset_name}: {e}")
        return None, 0

def balance_client_data(train_idx, num_clients, train_data, config):
    print(f"Начало balance_client_data с {num_clients} клиентами")
    print(f"train_data.x shape: {train_data.x.shape}")
    print(f"train_data.edge_index shape: {train_data.edge_index.shape}")
    print(f"train_data.y shape: {train_data.y.shape}")
    
    has_timestamps = hasattr(train_data, 'timestamps') and train_data.timestamps is not None
    has_transaction_types = hasattr(train_data, 'transaction_types') and train_data.transaction_types is not None
    
    max_edge_idx = train_data.edge_index.max().item()
    if max_edge_idx >= train_data.x.size(0):
        raise ValueError(f"Индекс ребра {max_edge_idx} вне диапазона для {train_data.x.size(0)} узлов")
    
    if isinstance(train_idx, np.ndarray):
        train_idx = torch.from_numpy(train_idx).long()
    
    valid_mask = (train_idx >= 0) & (train_idx < train_data.x.size(0))
    if not valid_mask.all():
        print(f"Предупреждение: Удалено {(~valid_mask).sum().item()} некорректных индексов из train_idx")
        train_idx = train_idx[valid_mask]
    
    y = train_data.y[train_idx].cpu().numpy()
    X = train_data.x[train_idx].cpu().numpy()
    timestamps = train_data.timestamps[train_idx].cpu().numpy()
    transaction_types = train_data.transaction_types[train_idx].cpu().numpy()
    
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=config["seed"])
    client_data = []
    client_sizes = []
    
    for _, client_idx in skf.split(X, y):
        client_indices = train_idx[client_idx]
        client_X = X[client_idx]
        client_y = y[client_idx]
        client_timestamps = timestamps[client_idx]
        client_transaction_types = transaction_types[client_idx]
        
        smote = SMOTE(random_state=config["seed"])
        client_X, client_y = smote.fit_resample(client_X, client_y)
        client_timestamps = np.repeat(client_timestamps, len(client_X) // len(client_timestamps) + 1)[:len(client_X)]
        client_transaction_types = np.repeat(client_transaction_types, len(client_X) // len(client_transaction_types) + 1)[:len(client_X)]
        
        knn = NearestNeighbors(n_neighbors=config["n_neighbors"], metric="cosine", n_jobs=-1)
        knn.fit(client_X)
        distances, indices = knn.kneighbors(client_X)
        
        edge_index = []
        edge_attr = []
        for i in range(len(client_X)):
            neighbors = indices[i]
            for j in neighbors:
                if i != j and j < len(client_X):
                    edge_index.append((i, j))
                    time_diff = client_timestamps[j] - client_timestamps[i]
                    edge_attr.append(time_diff / (np.std(client_timestamps) + 1e-9))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(train_data.x.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(train_data.x.device)
        
        client_data.append(Data(
            x=torch.FloatTensor(client_X).to(train_data.x.device),
            y=torch.FloatTensor(client_y).to(train_data.y.device),
            edge_index=edge_index.to(train_data.x.device),
            edge_attr=edge_attr.to(train_data.x.device) if edge_attr.numel() > 0 else None,
            transaction_types=torch.LongTensor(client_transaction_types).to(train_data.x.device),
            timestamps=torch.FloatTensor(client_timestamps).to(train_data.x.device)
        ))
        client_sizes.append(len(client_X))
        print(f"Клиент {len(client_data)-1} данные созданы с {client_sizes[-1]} узлами, классы: {np.bincount(client_y.astype(int))}")
    
    print(f"Завершена балансировка данных для {num_clients} клиентов")
    return client_data, client_sizes

def main():
    config = {
        "seed": 42,
        "hidden_dim": 128,
        "output_dim": 1,
        "num_transaction_types": 10,
        "num_clients": 5,
        "num_epochs": 50,
        "n_splits": 5,
        "k_percent": 1,
        "tssgc_lr": 0.001,
        "dqn_lr": 0.001,
        "replay_buffer_size": 10000,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
        "batch_size": 64,
        "threshold_min": 0.1,
        "threshold_max": 0.9,
        "num_thresholds": 10,
        "cache_dir": "cache",
        "max_rows": 10000,
        "n_neighbors": 10,
        "top_features": None,
        "time_step": 3600,
        "early_stopping_patience": 5,
        "early_stopping_delta": 0.005,
        "noise_level": 0.5,
        "dqn_hidden_dim": 128,
        "max_edges_per_client": 5000,
        "local_paths": {
            "credit_card_2023": None,
            "financial_fraud": None
        }
    }
    
    device, gpu_name, total_memory = check_gpu()
    set_seed(config["seed"])
    
    dataset_configs = {
        "credit_card_2023": "nelgiriyewithana/credit-card-fraud-detection-dataset-2023",
        "financial_fraud": "sriharshaeedala/financial-fraud-detection-dataset"
    }
    
    for dataset_name, dataset_slug in dataset_configs.items():
        print(f"\nОбработка датасета: {dataset_name}")
        file_path = download_kaggle_dataset(dataset_slug, config["cache_dir"], config["local_paths"].get(dataset_name))
        if not file_path:
            print(f"Не удалось загрузить {dataset_name}. Пропуск.")
            continue
        
        data, input_dim = load_and_preprocess_data(dataset_name, file_path, config)
        if data is None:
            print(f"Не удалось предобработать {dataset_name}. Пропуск.")
            continue
        
        class_counts = np.bincount(data.y.cpu().numpy().astype(int))
        print(f"Распределение классов после предобработки: {class_counts}")
        class_weights = torch.FloatTensor([1.0 / max(class_counts[1], 1) if i == 1 else 1.0 / max(class_counts[0], 1) for i in range(2)]).to(device)
        class_weights = class_weights / class_weights.sum() * 2
        
        action_space_size = config["num_thresholds"]
        print(f"Размер пространства действий: {action_space_size}")
        
        skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
        fold_results = {"auc_roc": [], "auc_pr": [], "f1": [], "recall_at_k": [], "adv_f1": []}
        fold_metrics = []
        
        print(f"Начало обучения на {dataset_name} с {gpu_name if gpu_name else 'CPU'}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(data.x.size(0)), data.y.cpu().numpy())):
            print(f"\nФолд {fold+1}/{config['n_splits']}")
            train_edge_index, train_edge_attr = filter_and_reindex_edges(data.edge_index.cpu(), data.edge_attr.cpu(), train_idx)
            val_edge_index, val_edge_attr = filter_and_reindex_edges(data.edge_index.cpu(), data.edge_attr.cpu(), val_idx)
            
            train_data = Data(
                x=data.x[train_idx].to(device),
                edge_index=train_edge_index.to(device),
                edge_attr=train_edge_attr.to(device) if train_edge_attr is not None else None,
                transaction_types=data.transaction_types[train_idx].to(device),
                timestamps=data.timestamps[train_idx].to(device),
                y=data.y[train_idx].to(device)
            )
            val_data = Data(
                x=data.x[val_idx].to(device),
                edge_index=val_edge_index.to(device),
                edge_attr=val_edge_attr.to(device) if val_edge_attr is not None else None,
                transaction_types=data.transaction_types[val_idx].to(device),
                timestamps=data.timestamps[val_idx].to(device),
                y=data.y[val_idx].to(device)
            )
            
            client_data, client_sizes = balance_client_data(train_idx, config["num_clients"], train_data, config)
            
            server_model = FraudGNNRL(input_dim, config["hidden_dim"], config["output_dim"], config["num_transaction_types"], action_space_size, config)
            clients = [FraudGNNRL(input_dim, config["hidden_dim"], config["output_dim"], config["num_transaction_types"], action_space_size, config) for _ in range(config["num_clients"])]
            server = FederatedServer(server_model.tssgc, server_model.dqn, config["num_clients"])
            early_stopping = EarlyStopping(config["early_stopping_patience"], config["early_stopping_delta"])
            
            for client in clients:
                for _ in range(50):
                    state = np.zeros(input_dim + 4)
                    action = random.randrange(action_space_size)
                    reward = 0.0
                    next_state = state
                    done = 0
                    client.store_transition(state, action, reward, next_state, done)
            
            epoch_metrics = {"auc_roc": [], "auc_pr": [], "f1": [], "recall_at_k": [], "adv_f1": []}
            for epoch in range(config["num_epochs"]):
                start_time = time.time()
                print(f"Начало эпохи {epoch+1}...")
                
                client_tssgcs = []
                client_dqns = []
                for client_idx, (client, c_data) in enumerate(zip(clients, client_data)):
                    client.tssgc.load_state_dict(server.global_tssgc.state_dict())
                    client.dqn.load_state_dict(server.global_dqn.state_dict())
                    
                    tssgc_start = time.time()
                    tssgc_loss = client.train_tssgc(c_data, c_data.y, class_weights[c_data.y.long()])
                    tssgc_time = time.time() - tssgc_start
                    
                    dqn_start = time.time()
                    degree = torch.bincount(c_data.edge_index[0], minlength=c_data.x.size(0)) + torch.bincount(c_data.edge_index[1], minlength=c_data.x.size(0))
                    avg_degree = degree.float().mean().item()
                    fraud_ratio = c_data.y.mean().item()
                    probs, predictions = client.predict(c_data, client.act(np.zeros(input_dim + 4)))
                    f1 = f1_score(c_data.y.cpu().numpy(), predictions.cpu().numpy(), zero_division=0)
                    auc_roc = roc_auc_score(c_data.y.cpu().numpy(), probs.cpu().numpy()) if len(np.unique(c_data.y.cpu().numpy())) > 1 else 0.5
                    state = torch.cat([c_data.x.mean(dim=0), torch.tensor([avg_degree, fraud_ratio, f1, auc_roc], device=device)]).detach().cpu().numpy()
                    action = client.act(state)
                    probs, predictions = client.predict(c_data, action)
                    f1 = f1_score(c_data.y.cpu().numpy(), predictions.cpu().numpy(), zero_division=0)
                    recall_k = recall_at_k(c_data.y.cpu().numpy(), probs.cpu().numpy(), k_percent=1)
                    reward = 0.4 * f1 + 0.6 * recall_k - 0.2 * (1 - recall_k)
                    next_fraud_ratio = c_data.y.mean().item()
                    next_f1 = f1
                    next_auc_roc = auc_roc
                    next_state = torch.cat([c_data.x.mean(dim=0), torch.tensor([avg_degree, next_fraud_ratio, next_f1, next_auc_roc], device=device)]).detach().cpu().numpy()
                    done = 0
                    client.store_transition(state, action, reward, next_state, done)
                    dqn_loss = client.train_dqn()
                    dqn_time = time.time() - dqn_start
                    
                    client_tssgcs.append(client.tssgc)
                    client_dqns.append(client.dqn)
                    print(f"Фолд {fold+1}, Эпоха {epoch+1}, Клиент {client_idx+1}: TSSGC Loss={tssgc_loss:.4f} ({tssgc_time:.2f}s), DQN Loss={dqn_loss:.4f} ({dqn_time:.2f}s), F1={f1:.4f}, Recall@1%={recall_k:.4f}, Размер буфера={len(client.replay_buffer)}")
                
                agg_start = time.time()
                server.aggregate(client_tssgcs, client_dqns, client_sizes)
                agg_time = time.time() - agg_start
                print(f"Фолд {fold+1}, Эпоха {epoch+1}: Агрегация завершена за {agg_time:.2f} секунд")
                
                for client in clients:
                    client.update_target_dqn()
                
                val_start = time.time()
                degree = torch.bincount(val_data.edge_index[0], minlength=val_data.x.size(0)) + torch.bincount(val_data.edge_index[1], minlength=val_data.x.size(0))
                avg_degree = degree.float().mean().item()
                fraud_ratio = val_data.y.mean().item()
                probs, predictions = server_model.predict(val_data, server_model.act(np.zeros(input_dim + 4)))
                val_f1 = f1_score(val_data.y.cpu().numpy(), predictions.cpu().numpy(), zero_division=0)
                val_auc_roc = roc_auc_score(val_data.y.cpu().numpy(), probs.cpu().numpy()) if len(np.unique(val_data.y.cpu().numpy())) > 1 else 0.5
                state = torch.cat([val_data.x.mean(dim=0), torch.tensor([avg_degree, fraud_ratio, val_f1, val_auc_roc], device=device)]).detach().cpu().numpy()
                action = server_model.act(state)
                probs, predictions = server_model.predict(val_data, action)
                val_f1 = f1_score(val_data.y.cpu().numpy(), predictions.cpu().numpy(), zero_division=0)
                val_auc_roc = roc_auc_score(val_data.y.cpu().numpy(), probs.cpu().numpy()) if len(np.unique(val_data.y.cpu().numpy())) > 1 else 0.5
                val_auc_pr = average_precision_score(val_data.y.cpu().numpy(), probs.cpu().numpy())
                val_recall_at_k = recall_at_k(val_data.y.cpu().numpy(), probs.cpu().numpy(), config["k_percent"])
                
                adv_metrics = evaluate_adversarial_robustness(server_model, val_data, action, config["noise_level"])
                val_time = time.time() - val_start
                print(f"Фолд {fold+1}, Эпоха {epoch+1}, Валидация: AUC-ROC={val_auc_roc:.4f}, AUC-PR={val_auc_pr:.4f}, F1={val_f1:.4f}, Recall@{config['k_percent']}%={val_recall_at_k:.4f}, Adv F1={adv_metrics['f1']:.4f} ({val_time:.2f}s)")
                
                epoch_metrics["auc_roc"].append(val_auc_roc)
                epoch_metrics["auc_pr"].append(val_auc_pr)
                epoch_metrics["f1"].append(val_f1)
                epoch_metrics["recall_at_k"].append(val_recall_at_k)
                epoch_metrics["adv_f1"].append(adv_metrics["f1"])
                
                early_stopping(val_f1, server_model.tssgc)
                if early_stopping.early_stop:
                    print(f"Ранняя остановка на эпохе {epoch+1}.")
                    server_model.tssgc.load_state_dict(early_stopping.best_model_state)
                    torch.save(early_stopping.best_model_state, f"{dataset_name}_fold{fold+1}_best_model.pth")
                    break
                
                epoch_time = time.time() - start_time
                print(f"Эпоха {epoch+1} завершена за {epoch_time:.2f} секунд")
            
            fold_results["auc_roc"].append(val_auc_roc)
            fold_results["auc_pr"].append(val_auc_pr)
            fold_results["f1"].append(val_f1)
            fold_results["recall_at_k"].append(val_recall_at_k)
            fold_results["adv_f1"].append(adv_metrics["f1"])
            fold_metrics.append(epoch_metrics)
        
        print(f"\nDataset {dataset_name} Cross-validation results:")
        print(f"Mean AUC-ROC = {np.mean(fold_results['auc_roc']):.4f}, Std = {np.std(fold_results['auc_roc']):.4f}")
        print(f"Mean AUC-PR = {np.mean(fold_results['auc_pr']):.4f}, Std = {np.std(fold_results['auc_pr']):.4f}")
        print(f"Mean F1 = {np.mean(fold_results['f1']):.4f}, Std = {np.std(fold_results['f1']):.4f}")
        print(f"Mean Recall@{config['k_percent']}% = {np.mean(fold_results['recall_at_k']):.4f}, Std = {np.std(fold_results['recall_at_k']):.4f}")
        print(f"Mean Adversarial F1 = {np.mean(fold_results['adv_f1']):.4f}, Std = {np.std(fold_results['adv_f1']):.4f}")
        
        epochs = list(range(1, len(fold_metrics[0]["auc_roc"]) + 1))
        auc_roc_data = [np.mean([fold_metrics[i]["auc_roc"][j] for i in range(config["n_splits"]) if j < len(fold_metrics[i]["auc_roc"])]) for j in range(len(fold_metrics[0]["auc_roc"]))]
        auc_pr_data = [np.mean([fold_metrics[i]["auc_pr"][j] for i in range(config["n_splits"]) if j < len(fold_metrics[i]["auc_pr"])]) for j in range(len(fold_metrics[0]["auc_pr"]))]
        f1_data = [np.mean([fold_metrics[i]["f1"][j] for i in range(config["n_splits"]) if j < len(fold_metrics[i]["f1"])]) for j in range(len(fold_metrics[0]["f1"]))]
        recall_data = [np.mean([fold_metrics[i]["recall_at_k"][j] for i in range(config["n_splits"]) if j < len(fold_metrics[i]["recall_at_k"])]) for j in range(len(fold_metrics[0]["recall_at_k"]))]
        adv_f1_data = [np.mean([fold_metrics[i]["adv_f1"][j] for i in range(config["n_splits"]) if j < len(fold_metrics[i]["adv_f1"])]) for j in range(len(fold_metrics[0]["adv_f1"]))]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, auc_roc_data, label="AUC-ROC", color="blue")
        plt.plot(epochs, auc_pr_data, label="AUC-PR", color="orange")
        plt.plot(epochs, f1_data, label="F1-Score", color="green")
        plt.plot(epochs, recall_data, label=f"Recall@{config['k_percent']}%", color="red")
        plt.plot(epochs, adv_f1_data, label="Adversarial F1", color="purple")
        plt.xlabel("Эпоха")
        plt.ylabel("Значение метрики")
        plt.title(f"Метрики валидации по эпохам для {dataset_name}")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.savefig(f"{dataset_name}_metrics.png")
        plt.close()

if __name__ == "__main__":
    main()