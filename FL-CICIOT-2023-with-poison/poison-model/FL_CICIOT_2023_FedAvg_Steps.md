# Federated Learning with CIC-IoT-2023 Dataset - Step-by-Step Guide

## Overview
This notebook implements Federated Learning using the FedAvg algorithm on the CIC-IoT-2023 dataset for IoT attack detection.

## Step 1: Environment Setup

### 1.1 Install Dependencies
```bash
pip install flwr[simulation] torch torchvision matplotlib scikit-learn openml
```

### 1.2 Import Required Libraries
```python
import os
import pandas as pd
import numpy as np
import flwr as fl
from tqdm import tqdm
import warnings
from typing import List, Tuple, Optional, Dict, Union
from scipy.spatial.distance import cosine

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import Metrics, Parameters, Scalar
from torch.utils.data import DataLoader, random_split, TensorDataset
```

### 1.3 Check Environment
```python
print("flwr", fl.__version__)
print("numpy", np.__version__)
print("torch", torch.__version__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
```

## Step 2: Configuration Settings

### 2.1 Federated Learning Configuration
```python
SPLIT_AVAILABLE_METHODS = ['STRATIFIED','LEAVE_ONE_OUT', 'ONE_CLASS', 'HALF_BENIGN' ,'DIRICHLET']
METHOD = 'DIRICHLET'
NUM_OF_STRATIFIED_CLIENTS = 10  # only applies to stratified method
NUM_OF_ROUNDS = 5              # Number of FL rounds
```

### 2.2 Classification Type Configuration
```python
individual_classifier = True   # 34-class classification
group_classifier = False      # 8-class classification
binary_classifier = False     # 2-class classification
```

### 2.3 Defense Strategy Selection
```python
# Choose ONE defense strategy at a time
DEFENSE_STRATEGY = 'FLTRUST'  # Options: 'NONE', 'KRUM', 'TRIMMED_MEAN', 'MEDIAN', 'ANOMALY_DETECTION', 'FLTRUST'

# Defense parameters (only used when defense is enabled)
ANOMALY_THRESHOLD = 0.7
TRIM_RATIO = 0.2
BYZANTINE_CLIENTS = 2
FLTRUST_BETA = 0.5  # Trust score threshold for FLTrust
SERVER_DATA_RATIO = 0.1  # Ratio of server data for FLTrust
KRUM_M = 2  # Number of Byzantine clients for Krum

print(f"Selected defense strategy: {DEFENSE_STRATEGY}")
```

## Step 3: Data Access Setup

### 3.1 Mount Google Drive (for Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3.2 Define Dataset Directory
```python
DATASET_DIRECTORY = '/content/drive/MyDrive/Colab Notebooks/data/CICIoT2023/'
```

## Step 4: Data Loading and Preprocessing

### 4.1 Load Dataset Files
```python
# Load all CSV files
df_sets = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
df_sets.sort()

# Use 80% for training
training_sets = df_sets[:int(len(df_sets)*.8)]
```

### 4.2 Data Exploration
```python
# Check dataset structure
sample_df = pd.read_csv(DATASET_DIRECTORY + training_sets[0])
print(f"Available columns in dataset: {list(sample_df.columns)}")
print(f"Dataset shape: {sample_df.shape}")
```

### 4.3 Data Combination and Preprocessing
```python
# Combine all training data with immediate rounding
combined_df = pd.DataFrame()
for file in tqdm(training_sets):
    df_temp = pd.read_csv(DATASET_DIRECTORY + file)
    
    # Round numbers for memory efficiency
    for col in df_temp.columns:
        if col != 'Label' and df_temp[col].dtype in ['float64', 'float32']:
            col_max = df_temp[col].abs().max()
            
            if col_max > 1000:
                df_temp[col] = df_temp[col].round(2)
            elif col_max > 1:
                df_temp[col] = df_temp[col].round(4)
            else:
                df_temp[col] = df_temp[col].round(6)
    
    combined_df = pd.concat([combined_df, df_temp], ignore_index=True)
```

## Step 5: Label Mapping and Classification Setup

### 5.1 Define Label Mappings
```python
# 34-class mapping (individual attacks)
dict_34_classes = {
    'BENIGN': 0, 'DDOS-RSTFINFLOOD': 1, 'DDOS-PSHACK_FLOOD': 2,
    # ... (complete mapping)
}

# 8-class mapping (attack groups)
dict_8_classes = {
    0: 0,  # Benign
    1:1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1,  # DDoS
    # ... (complete mapping)
}

# 2-class mapping (binary classification)
dict_2_classes = {
    0: 0,  # Benign
    # All attacks mapped to 1
}
```

### 5.2 Apply Label Mapping
```python
# Apply appropriate label mapping based on classification type
combined_df['Label'] = combined_df['Label'].map(dict_34_classes)

if group_classifier:
    combined_df['Label'] = combined_df['Label'].map(dict_8_classes)
elif binary_classifier:
    combined_df['Label'] = combined_df['Label'].map(dict_2_classes)

# Clean data
combined_df = combined_df.dropna(subset=['Label'])
combined_df['Label'] = combined_df['Label'].astype(int)
```

## Step 6: Training Data Preparation

### 6.1 Load or Create Training Data
```python
if os.path.isfile('training_data.pkl'):
    print("File exists, loading data...")
    train_df = pd.read_pickle('training_data.pkl')
else:
    # Process training data from CSV files
    dfs = []
    for train_set in tqdm(training_sets):
        df_new = pd.read_csv(DATASET_DIRECTORY + train_set)
        dfs.append(df_new)
    train_df = pd.concat(dfs, ignore_index=True)
```

### 6.2 Data Splitting for Memory Management
```python
TRAIN_SIZE = 0.99  # Use 99% of training data

X_train, X_test, y_train, y_test = train_test_split(
    train_df[X_columns], 
    train_df[y_column], 
    test_size=(1 - TRAIN_SIZE), 
    random_state=42, 
    stratify=train_df[y_column]
)

# Recombine into dataframe
train_df = pd.concat([X_train, y_train], axis=1)

# Save processed data
train_df.to_pickle('training_data.pkl')
```

### 6.3 Data Analysis
```python
print("Training data size: {}".format(train_df.shape))
print("Counts of attacks in train_df:")
print(train_df['Label'].value_counts())
```

## Step 7: Test Data Preparation

### 7.1 Load or Create Test Data
```python
# Check for existing test data pickle file
testing_data_pickle_file = 'testing_data.pkl'

if os.path.isfile(testing_data_pickle_file):
    print(f"File {testing_data_pickle_file} exists, loading data...")
    test_df = pd.read_pickle(testing_data_pickle_file)
    print("Test data loaded from pickle file.")
else:
    print(f"File {testing_data_pickle_file} does not exist, constructing data...")
    
    # Use remaining 20% of CSV files for testing
    test_sets = df_sets[int(len(df_sets)*.8):]
    
    # Load and combine test data
    dfs = []
    print("Reading test data...")
    for test_set in tqdm(test_sets):
        df_new = pd.read_csv(DATASET_DIRECTORY + test_set)
        dfs.append(df_new)
    test_df = pd.concat(dfs, ignore_index=True)
    
    # Apply label mapping
    test_df['Label'] = test_df['Label'].map(dict_34_classes)
    
    # Save processed test data
    print(f"Writing test data to pickle file {testing_data_pickle_file}...")
    test_df.to_pickle(testing_data_pickle_file)

print("Testing data size: {}".format(test_df.shape))
```

### 7.2 Data Size Comparison
```python
print("Number of rows in train_df: {}".format(len(train_df)))
print("Number of rows in test_df: {}".format(len(test_df)))

train_size = len(train_df)
test_size = len(test_df)
```

## Step 8: Data Scaling and Preprocessing

### 8.1 Scale Training Data
```python
scaler = StandardScaler()

# Handle infinite values and NaN
print("Checking for and handling infinite values...")
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.dropna(inplace=True)
print("Infinite values handled and rows with NaN removed.")

# Apply scaling to feature columns
train_df[X_columns] = scaler.fit_transform(train_df[X_columns])
```

### 8.2 Scale Test Data
```python
# Handle infinite values and NaN in test data
print("Checking for and handling infinite values in test data...")
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.dropna(inplace=True)
print("Infinite values handled and rows with NaN removed from test data.")

# Apply same scaling transformation to test data
test_df[X_columns] = scaler.transform(test_df[X_columns])
```

### 8.3 Classification Type Configuration
```python
class_size_map = {2: "Binary", 8: "Group", 34: "Individual"}

if group_classifier:
    print("Group 8 Class Classifier... - Adjusting labels in test and train dataframes")
    test_df['label'] = test_df['label'].map(dict_8_classes)
    train_df['label'] = train_df['label'].map(dict_8_classes)
    class_size = "8"
elif binary_classifier:
    print("Binary 2 Class Classifier... - Adjusting labels in test and train dataframes")
    test_df['label'] = test_df['label'].map(dict_2_classes)
    train_df['label'] = train_df['label'].map(dict_2_classes)
    class_size = "2"
else:
    print("Individual 34-class classification")
    class_size = "34"

print(f"Classification type: {class_size_map.get(int(class_size), 'Unknown')}")
```

## Step 9: Defense Implementation

### 9.1 FLTrust Defense
```python
class FLTrustDefense:
    def __init__(self, server_data_ratio=0.1, beta=0.5):
        self.server_data_ratio = server_data_ratio
        self.beta = beta
        self.server_model = None
        self.server_data = None
        
    def prepare_server_data(self, train_df, X_columns, y_column):
        """Prepare server dataset for FLTrust"""
        server_size = int(len(train_df) * self.server_data_ratio)
        self.server_data = train_df.sample(n=server_size, random_state=42)
        print(f"Server data size: {len(self.server_data)}")
        
    def compute_trust_scores(self, client_updates, server_update):
        """Compute trust scores based on cosine similarity"""
        trust_scores = []
        server_flat = np.concatenate([p.flatten() for p in server_update])
        
        for update in client_updates:
            client_flat = np.concatenate([p.flatten() for p in update])
            similarity = np.dot(server_flat, client_flat) / (np.linalg.norm(server_flat) * np.linalg.norm(client_flat))
            trust_scores.append(max(0, similarity))
            
        return np.array(trust_scores)
        
    def aggregate(self, client_updates, server_update):
        """FLTrust aggregation with trust scores"""
        trust_scores = self.compute_trust_scores(client_updates, server_update)
        
        # Normalize trust scores
        if trust_scores.sum() > 0:
            trust_scores = trust_scores / trust_scores.sum()
        else:
            trust_scores = np.ones(len(client_updates)) / len(client_updates)
            
        # Weighted aggregation
        aggregated = [np.zeros_like(param) for param in client_updates[0]]
        for i, update in enumerate(client_updates):
            for j, param in enumerate(update):
                aggregated[j] += trust_scores[i] * param
                
        return aggregated
```

### 9.2 Krum Defense
```python
class KrumDefense:
    def __init__(self, num_byzantine=2):
        self.num_byzantine = num_byzantine
        
    def compute_distances(self, client_updates):
        """Compute pairwise distances between client updates"""
        n_clients = len(client_updates)
        distances = np.zeros((n_clients, n_clients))
        
        for i in range(n_clients):
            for j in range(i+1, n_clients):
                # Flatten and compute L2 distance
                update_i = np.concatenate([p.flatten() for p in client_updates[i]])
                update_j = np.concatenate([p.flatten() for p in client_updates[j]])
                dist = np.linalg.norm(update_i - update_j)
                distances[i, j] = distances[j, i] = dist
                
        return distances
        
    def aggregate(self, client_updates):
        """Krum aggregation - select client with minimum score"""
        n_clients = len(client_updates)
        distances = self.compute_distances(client_updates)
        
        scores = []
        for i in range(n_clients):
            # Sum of distances to closest n-f-2 clients
            client_distances = distances[i]
            closest_distances = np.sort(client_distances)[1:n_clients-self.num_byzantine-1]
            scores.append(np.sum(closest_distances))
            
        # Return update from client with minimum score
        selected_client = np.argmin(scores)
        print(f"Krum selected client {selected_client}")
        return client_updates[selected_client]
```

## Step 10: Defense Configuration

### 10.1 Defense Factory
```python
def create_defense(defense_strategy):
    """Factory function to create defense instances"""
    if defense_strategy == 'FLTRUST':
        return FLTrustDefense(SERVER_DATA_RATIO, FLTRUST_BETA)
    elif defense_strategy == 'KRUM':
        return KrumDefense(KRUM_M)
    else:
        return None
        
defense_instance = create_defense(DEFENSE_STRATEGY)
print(f"Defense instance created: {type(defense_instance).__name__ if defense_instance else 'None'}")
```

### 10.2 Defense Integration
```python
if DEFENSE_STRATEGY == 'FLTRUST' and defense_instance:
    # Prepare server data for FLTrust
    defense_instance.prepare_server_data(train_df, X_columns, y_column)
    
    # Create server model for FLTrust
    server_model_fltrust = IoTAttackNet(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
    
    # Server training function
    def train_server_model():
        server_X = torch.FloatTensor(defense_instance.server_data[X_columns].values)
        server_y = torch.LongTensor(defense_instance.server_data[y_column].values)
        server_dataset = TensorDataset(server_X, server_y)
        server_loader = DataLoader(server_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        train_model(server_model_fltrust, server_loader, epochs=EPOCHS)
        return get_model_parameters(server_model_fltrust)
        
    print("FLTrust server model prepared")

print(f"Defense configuration completed for {DEFENSE_STRATEGY}")4 Class classifier... - No adjustments to labels in test and train dataframes")
    class_size = "34"
```

## Step 9: Federated Learning Data Distribution

### 9.1 Data Distribution Methods
```python
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Define available methods
SPLIT_AVAILABLE_METHODS = ['STRATIFIED','LEAVE_ONE_OUT', 'ONE_CLASS', 'HALF_BENIGN' ,'DIRICHLET']
METHOD = 'DIRICHLET'  # Current method

# Initialize client data containers
fl_X_train = []
fl_y_train = []
client_df = pd.DataFrame()
y_column = 'Label'
```

### 9.2 Stratified Distribution
```python
if METHOD == 'STRATIFIED':
    print(f"STRATIFIED METHOD with {class_size} class classifier")
    skf = StratifiedKFold(n_splits=NUM_OF_STRATIFIED_CLIENTS, shuffle=True, random_state=42)
    for _, test_index in skf.split(train_df[X_columns], train_df[y_column]):
        fl_X_train.append(train_df.iloc[test_index][X_columns])
        fl_y_train.append(train_df.iloc[test_index][y_column])
```

### 9.3 Leave-One-Out Distribution
```python
elif METHOD == 'LEAVE_ONE_OUT':
    print(f"LEAVE_ONE_OUT METHOD with {class_size} class classifier")
    
    num_splits = int(class_size) - 1 if (individual_classifier or group_classifier) else 10
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    for i, (_, test_index) in enumerate(skf.split(train_df[X_columns], train_df[y_column])):
        current_fold_df = train_df.iloc[test_index]
        if binary_classifier:
            # Even-indexed client: exclude attack class 1
            if i % 2 == 0:
                client_df = current_fold_df[current_fold_df[y_column] != 1].copy()
            else:
                client_df = current_fold_df.copy()
        else:
            # Exclude one specific attack class
            client_df = current_fold_df[current_fold_df[y_column] != (i + 1)].copy()
        
        fl_X_train.append(client_df[X_columns])
        fl_y_train.append(client_df[y_column])
```

### 9.4 One-Class Distribution
```python
elif METHOD == 'ONE_CLASS':
    print(f"ONE_CLASS METHOD with {class_size} class classifier")
    
    num_splits = int(class_size) - 1 if (individual_classifier or group_classifier) else 10
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    for i, (_, test_index) in enumerate(skf.split(train_df[X_columns], train_df[y_column])):
        current_fold_df = train_df.iloc[test_index]
        if binary_classifier:
            # Even-indexed client: only Benign data
            if i % 2 == 0:
                client_df = current_fold_df[current_fold_df[y_column] != 1].copy()
            else:
                client_df = current_fold_df.copy()
        else:
            # Include only Benign and the (i+1)-th attack class
            mask = (current_fold_df[y_column] == 0) | (current_fold_df[y_column] == (i + 1))
            client_df = current_fold_df[mask].copy()
        
        fl_X_train.append(client_df[X_columns])
        fl_y_train.append(client_df[y_column])
```

### 9.5 Half-Benign Distribution
```python
elif METHOD == 'HALF_BENIGN':
    print(f"HALF_BENIGN METHOD with {class_size} class classifier")
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for i, (_, test_index) in enumerate(skf.split(train_df[X_columns], train_df[y_column])):
        current_fold_df = train_df.iloc[test_index]
        if i % 2 == 0:
            # Even-indexed clients: only Benign data
            client_df = current_fold_df[current_fold_df[y_column] == 0].copy()
        else:
            # Odd-indexed clients: all data
            client_df = current_fold_df.copy()
        
        fl_X_train.append(client_df[X_columns])
        fl_y_train.append(client_df[y_column])
```

### 9.6 Dirichlet Distribution (Non-IID)
```python
elif METHOD == 'DIRICHLET':
    print(f"DIRICHLET METHOD with {class_size} class classifier")
    
    # Number of clients
    num_clients = NUM_OF_DIRICHLET_CLIENTS if 'NUM_OF_DIRICHLET_CLIENTS' in globals() else 10
    
    # Alpha parameter controls non-IID level (smaller = more non-IID)
    alpha = DIRICHLET_ALPHA if 'DIRICHLET_ALPHA' in globals() else 0.5
    
    # Get data arrays
    y_data = train_df[y_column].values
    X_data = train_df[X_columns].values
    
    # Split sample indices by class
    class_indices = {}
    for cls in np.unique(y_data):
        class_indices[cls] = np.where(y_data == cls)[0]
    
    # Create Dirichlet distribution for each class
    client_indices = [[] for _ in range(num_clients)]
    for cls, indices in class_indices.items():
        n_samples = len(indices)
        
        # Generate Dirichlet proportions
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Calculate samples per client
        splits = (proportions * n_samples).astype(int)
        
        # Handle rounding errors
        while splits.sum() < n_samples:
            splits[np.argmax(proportions)] += 1
        while splits.sum() > n_samples:
            splits[np.argmax(splits)] -= 1
        
        # Distribute samples to clients
        np.random.shuffle(indices)
        start = 0
        for client_id, split_size in enumerate(splits):
            end = start + split_size
            client_indices[client_id].extend(indices[start:end])
            start = end
    
    # Create client datasets
    for client_id in range(num_clients):
        idxs = client_indices[client_id]
        client_df = train_df.iloc[idxs]
        fl_X_train.append(client_df[X_columns])
        fl_y_train.append(client_df[y_column])
```

## Step 10: Poison Attack Implementation and Defense Implementation

### 10.1.1 Attack Configuration
```python
# General attack configuration
num_malicious_clients = 11  # about 33% of clients
malicious_client_ids = [str(i) for i in range(num_malicious_clients)]

# Attack types available
ATTACK_TYPES = [
    'MODEL_POISONING',     # Scale model parameters
    'MIMIC_ATTACK',        # Mimic benign behavior while being malicious
    'LABEL_FLIPPING',      # Flip labels during training
    'GRADIENT_ASCENT',     # Reverse gradient direction
    'BACKDOOR_ATTACK',     # Insert backdoor triggers
    'BYZANTINE_ATTACK',    # Send random/adversarial updates
    'DATA_POISONING',      # Poison training data
    'NONE'                 # No attack (baseline)
]

# Select attack type
ATTACK_TYPE = 'MODEL_POISONING'  # Change this to use different attacks

print(f"--- Configuring {ATTACK_TYPE} ---")
print(f"Malicious clients: {num_malicious_clients} ({malicious_client_ids})")
```

### 10.1.2 Model Poisoning Attack
```python
if ATTACK_TYPE == 'MODEL_POISONING':
    # Scale model parameters to overpower honest clients
    poison_factor = 3.0
    
    def apply_model_poisoning(model_params):
        """Scale model parameters by poison factor"""
        return [param * poison_factor for param in model_params]
    
    print(f"Model Poisoning - Scaling factor: {poison_factor}")
```

### 10.1.3 Mimic Attack
```python
elif ATTACK_TYPE == 'MIMIC_ATTACK':
    # Mimic benign behavior while being subtly malicious
    mimic_target_label = 0  # Target to misclassify as BENIGN
    mimic_strength = 1.5    # Subtle parameter modification
    mimic_noise_std = 0.1   # Add noise to hide attack
    
    def apply_mimic_attack(model_params):
        """Apply subtle parameter modifications with noise"""
        poisoned_params = []
        for param in model_params:
            # Add subtle scaling and noise
            noise = np.random.normal(0, mimic_noise_std, param.shape)
            poisoned_param = param * mimic_strength + noise
            poisoned_params.append(poisoned_param)
        return poisoned_params
    
    print(f"Mimic Attack - Target: {mimic_target_label}, Strength: {mimic_strength}")
```

### 10.1.4 Label Flipping Attack
```python
elif ATTACK_TYPE == 'LABEL_FLIPPING':
    # Flip labels during training
    flip_probability = 0.3  # Probability of flipping a label
    target_classes = [1, 2, 3]  # Classes to flip to benign (0)
    
    def apply_label_flipping(y_train):
        """Flip labels with given probability"""
        y_poisoned = y_train.copy()
        for i in range(len(y_poisoned)):
            if np.random.random() < flip_probability:
                if y_poisoned[i] in target_classes:
                    y_poisoned[i] = 0  # Flip to benign
        return y_poisoned
    
    print(f"Label Flipping - Probability: {flip_probability}, Target classes: {target_classes}")
```

### 10.1.5 Gradient Ascent Attack
```python
elif ATTACK_TYPE == 'GRADIENT_ASCENT':
    # Reverse gradient direction to maximize loss
    ascent_factor = -1.0  # Reverse gradient direction
    
    def apply_gradient_ascent(model_params, original_params):
        """Reverse gradient direction"""
        poisoned_params = []
        for param, orig_param in zip(model_params, original_params):
            gradient = param - orig_param
            poisoned_param = orig_param + (ascent_factor * gradient)
            poisoned_params.append(poisoned_param)
        return poisoned_params
    
    print(f"Gradient Ascent - Factor: {ascent_factor}")
```

### 10.1.6 Backdoor Attack
```python
elif ATTACK_TYPE == 'BACKDOOR_ATTACK':
    # Insert backdoor trigger in data
    backdoor_trigger_value = 999.0  # Trigger value
    backdoor_target_label = 0       # Target label (benign)
    backdoor_feature_idx = 0        # Feature index to modify
    backdoor_probability = 0.1      # Probability of inserting trigger
    
    def apply_backdoor_trigger(X_train, y_train):
        """Insert backdoor triggers in training data"""
        X_poisoned = X_train.copy()
        y_poisoned = y_train.copy()
        
        for i in range(len(X_poisoned)):
            if np.random.random() < backdoor_probability:
                X_poisoned.iloc[i, backdoor_feature_idx] = backdoor_trigger_value
                y_poisoned.iloc[i] = backdoor_target_label
        
        return X_poisoned, y_poisoned
    
    print(f"Backdoor Attack - Trigger: {backdoor_trigger_value}, Target: {backdoor_target_label}")
```

### 10.1.7 Byzantine Attack
```python
elif ATTACK_TYPE == 'BYZANTINE_ATTACK':
    # Send random or adversarial model updates
    byzantine_noise_scale = 10.0  # Scale of random noise
    
    def apply_byzantine_attack(model_params):
        """Generate random adversarial updates"""
        poisoned_params = []
        for param in model_params:
            # Generate random noise with same shape
            random_noise = np.random.normal(0, byzantine_noise_scale, param.shape)
            poisoned_params.append(random_noise.astype(param.dtype))
        return poisoned_params
    
    print(f"Byzantine Attack - Noise scale: {byzantine_noise_scale}")
```

### 10.1.8 Data Poisoning Attack
```python
elif ATTACK_TYPE == 'DATA_POISONING':
    # Poison training data by adding adversarial samples
    poison_ratio = 0.2          # Ratio of data to poison
    noise_magnitude = 0.5       # Magnitude of adversarial noise
    target_flip_label = 0       # Label to flip poisoned samples to
    
    def apply_data_poisoning(X_train, y_train):
        """Add adversarial noise to training data"""
        X_poisoned = X_train.copy()
        y_poisoned = y_train.copy()
        
        n_poison = int(len(X_train) * poison_ratio)
        poison_indices = np.random.choice(len(X_train), n_poison, replace=False)
        
        for idx in poison_indices:
            # Add adversarial noise
            noise = np.random.normal(0, noise_magnitude, X_train.shape[1])
            X_poisoned.iloc[idx] += noise
            y_poisoned.iloc[idx] = target_flip_label
        
        return X_poisoned, y_poisoned
    
    print(f"Data Poisoning - Ratio: {poison_ratio}, Noise: {noise_magnitude}")
```

### 10.1.9 No Attack (Baseline)
```python
else:
    # No attack - baseline federated learning
    ATTACK_TYPE = 'NONE'
    
    def apply_no_attack(model_params):
        """Return parameters unchanged"""
        return model_params
    
    print("No Attack - Baseline federated learning")
```

### 10.1.10 Attack Summary
```python
print(f"\n=== Attack Configuration Summary ===")
print(f"Attack Type: {ATTACK_TYPE}")
print(f"Malicious Clients: {num_malicious_clients}/{len(fl_X_train) if 'fl_X_train' in globals() else 'TBD'}")
print(f"Malicious Client IDs: {malicious_client_ids}")
print(f"Classification: {class_size}-class")
print("=" * 40)

# Store attack configuration for later use
attack_config = {
    'type': ATTACK_TYPE,
    'malicious_clients': malicious_client_ids,
    'num_malicious': num_malicious_clients
}
```
### 10.2.1 Defense Configuration
### 10.2. Defense Implement: FLTrust, Krum


## Step 11: Client Data Analysis



### 11.1 Update Client Count and Inspect Data
```python
# Update the number of clients created
NUM_OF_CLIENTS = len(fl_X_train)

# Inspect training data for each client
for i in range(NUM_OF_CLIENTS):
    print(f"\n--- Client ID: {i} ---")
    print(f"fl_X_train[{i}].shape: {fl_X_train[i].shape}")
    print(f"fl_y_train[{i}].value_counts():\n{fl_y_train[i].value_counts()}")
    print(f"fl_y_train[{i}].unique(): {fl_y_train[i].unique()}")

# Check if two clients have identical feature data
print(f"\nfl_X_train[0].equals(fl_X_train[1]): {fl_X_train[0].equals(fl_X_train[1])}")
```

## Step 12: Data Visualization

### 12.1 Visualize Data Distribution
```python
# Create visualization for client data distribution
import matplotlib.pyplot as plt
import seaborn as sns

# Set up color palette
colors1 = plt.cm.get_cmap('tab20', 20)

# Create distribution plots for different methods
# (Implementation depends on the specific method used)
```

## Key Features

### Dataset Characteristics
- **Source**: CIC-IoT-2023 Dataset
- **Attack Types**: 34 individual attack classes
- **Features**: 39 network traffic features
- **Size**: ~2.8M samples after preprocessing
- **Train/Test Split**: 80/20 split with further 99% sampling for memory management

### Federated Learning Setup
- **Algorithm**: FedAvg (Federated Averaging)
- **Framework**: Flower (flwr)
- **Data Distribution**: Multiple methods (Stratified, Dirichlet, Leave-One-Out, etc.)
- **Clients**: Variable number based on distribution method
- **Rounds**: Configurable (default: 5)

### Classification Options
1. **Individual Classifier**: 34-class classification (all attack types)
2. **Group Classifier**: 8-class classification (attack categories)
3. **Binary Classifier**: 2-class classification (benign vs malicious)

### Security Research Features
- **Poison Attacks**: Model poisoning and mimic attacks
- **Malicious Clients**: Configurable percentage of compromised clients
- **Attack Parameters**: Customizable poison factors and target labels

### Memory Optimization
- Numerical precision reduction for large datasets
- Pickle file caching for processed data
- Stratified sampling for manageable data sizes
- Infinite value handling and NaN removal

### Data Distribution Strategies
1. **Stratified**: Even distribution across clients
2. **Leave-One-Out**: Each client missing one attack class
3. **One-Class**: Each client has benign + one attack class
4. **Half-Benign**: Alternating benign-only and full-data clients
5. **Dirichlet**: Non-IID distribution with configurable alpha parameter

## Implementation Notes
- **Environment**: Designed for Google Colab with GPU support
- **Scalability**: Handles large datasets with memory optimization
- **Flexibility**: Multiple configuration options for different research scenarios
- **Security Focus**: Includes adversarial federated learning components
- **Preprocessing**: Comprehensive data cleaning and standardization

## Step 13: Neural Network Model Definition

### 13.1 Define PyTorch Model Architecture
```python
class IoTAttackNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IoTAttackNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Model parameters
INPUT_SIZE = len(X_columns)  # 39 features
NUM_CLASSES = int(class_size)  # 34, 8, or 2 classes
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 32

# Initialize model
model = IoTAttackNet(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

### 13.2 Model Utility Functions
```python
def get_model_parameters(model):
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """Set model parameters from a list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def train_model(model, trainloader, epochs=1):
    """Train the model on the training set."""
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model

def test_model(model, testloader):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    return test_loss, accuracy
```

## Step 13.5: Individual Defense Strategies

### 13.5.1 Krum Defense
```python
class KrumDefense:
    def __init__(self, num_byzantine=2):
        self.num_byzantine = num_byzantine
    
    def aggregate(self, client_weights, client_sizes):
        n_clients = len(client_weights)
        scores = []
        for i, weights_i in enumerate(client_weights):
            distances = [np.linalg.norm(weights_i - weights_j) 
                        for j, weights_j in enumerate(client_weights) if i != j]
            distances.sort()
            scores.append(sum(distances[:n_clients - self.num_byzantine - 2]))
        return client_weights[np.argmin(scores)]
```

### 13.5.2 Trimmed Mean Defense
```python
class TrimmedMeanDefense:
    def __init__(self, trim_ratio=0.2):
        self.trim_ratio = trim_ratio
    
    def aggregate(self, client_weights, client_sizes):
        stacked_weights = np.stack(client_weights)
        trim_count = int(len(client_weights) * self.trim_ratio)
        sorted_weights = np.sort(stacked_weights, axis=0)
        return np.mean(sorted_weights[trim_count:-trim_count or None], axis=0)
```

### 13.5.3 Median Defense
```python
class MedianDefense:
    def aggregate(self, client_weights, client_sizes):
        return np.median(np.stack(client_weights), axis=0)
```

### 13.5.4 Anomaly Detection Defense
```python
class AnomalyDetectionDefense:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.historical_updates = []
    
    def detect_anomalous_clients(self, client_updates):
        if len(self.historical_updates) == 0:
            self.historical_updates = client_updates.copy()
            return []
        
        avg_historical = np.mean(self.historical_updates, axis=0)
        anomalous_clients = []
        
        for i, update in enumerate(client_updates):
            similarity = 1 - cosine(update.flatten(), avg_historical.flatten())
            if similarity < self.threshold:
                anomalous_clients.append(i)
        
        return anomalous_clients
```

### 13.5.5 FLTrust Defense
```python
class FLTrustDefense:
    def __init__(self, beta=0.5, server_data=None):
        self.beta = beta  # Trust score threshold
        self.server_data = server_data
        self.server_update = None
    
    def set_server_update(self, server_update):
        """Set server update as reference"""
        self.server_update = server_update
    
    def compute_trust_scores(self, client_updates):
        """Compute trust scores based on cosine similarity with server update"""
        if self.server_update is None:
            return np.ones(len(client_updates))  # Equal trust if no server update
        
        trust_scores = []
        server_flat = self.server_update.flatten()
        
        for update in client_updates:
            client_flat = update.flatten()
            # Cosine similarity as trust score
            similarity = np.dot(server_flat, client_flat) / \
                        (np.linalg.norm(server_flat) * np.linalg.norm(client_flat))
            trust_score = max(0, similarity)  # Ensure non-negative
            trust_scores.append(trust_score)
        
        return np.array(trust_scores)
    
    def aggregate(self, client_updates, client_sizes):
        """Aggregate using trust scores"""
        trust_scores = self.compute_trust_scores(client_updates)
        
        # Filter clients with trust score above threshold
        trusted_indices = trust_scores >= self.beta
        
        if not np.any(trusted_indices):
            # If no trusted clients, use server update or fallback to FedAvg
            if self.server_update is not None:
                return self.server_update
            else:
                return np.mean(client_updates, axis=0)
        
        # Weighted aggregation with trust scores
        trusted_updates = [client_updates[i] for i in range(len(client_updates)) if trusted_indices[i]]
        trusted_scores = trust_scores[trusted_indices]
        trusted_sizes = [client_sizes[i] for i in range(len(client_sizes)) if trusted_indices[i]]
        
        # Normalize trust scores
        normalized_scores = trusted_scores / np.sum(trusted_scores)
        
        # Weighted aggregation
        aggregated = np.zeros_like(trusted_updates[0])
        for update, score in zip(trusted_updates, normalized_scores):
            aggregated += update * score
        
        return aggregated
```

## Step 14: Flower Client Implementation

### 14.1 Define Flower Client Class
```python
import flwr as fl
from flwr.client import NumPyClient
from torch.utils.data import TensorDataset, DataLoader

class IoTClient(NumPyClient):
    def __init__(self, client_id, X_train, y_train, X_test, y_test, is_malicious=False):
        self.client_id = client_id
        self.is_malicious = is_malicious
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train.values)
        self.y_train = torch.LongTensor(y_train.values)
        self.X_test = torch.FloatTensor(X_test.values)
        self.y_test = torch.LongTensor(y_test.values)
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        
        self.trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        self.model = IoTAttackNet(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    
    def get_parameters(self, config):
        """Return current model parameters."""
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        set_model_parameters(self.model, parameters)
    
    def fit(self, parameters, config):
        """Train the model with the given parameters."""
        self.set_parameters(parameters)
        
        # Normal training
        train_model(self.model, self.trainloader, epochs=EPOCHS)
        
        # Apply poison attack if malicious client
        if self.is_malicious:
            self.apply_poison_attack()
        
        return get_model_parameters(self.model), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model with the given parameters."""
        self.set_parameters(parameters)
        loss, accuracy = test_model(self.model, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}
    
    def apply_poison_attack(self):
        """Apply model poisoning attack."""
        if self.client_id in malicious_client_ids:
            # Scale model parameters by poison factor
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data *= poison_factor
            print(f"Client {self.client_id}: Applied poison attack with factor {poison_factor}")
```

### 14.2 Client Factory Function
```python
def client_fn(cid: str) -> IoTClient:
    """Create a Flower client representing a single organization."""
    client_id = int(cid)
    
    # Get client data
    X_train_client = fl_X_train[client_id]
    y_train_client = fl_y_train[client_id]
    
    # Use a portion of test data for client evaluation
    test_size_per_client = len(test_df) // NUM_OF_CLIENTS
    start_idx = client_id * test_size_per_client
    end_idx = start_idx + test_size_per_client
    
    X_test_client = test_df[X_columns].iloc[start_idx:end_idx]
    y_test_client = test_df[y_column].iloc[start_idx:end_idx]
    
    # Check if client is malicious
    is_malicious = cid in malicious_client_ids
    
    return IoTClient(client_id, X_train_client, y_train_client, 
                    X_test_client, y_test_client, is_malicious)
```

## Step 15: Server Strategy and Model Initialization

### 15.1 Initialize Server Model
```python
# Create initial model for server
server_model = IoTAttackNet(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
initial_parameters = get_model_parameters(server_model)

print(f"Server model initialized with {sum(p.numel() for p in server_model.parameters())} parameters")
print(f"Model architecture: {INPUT_SIZE} -> 128 -> 64 -> 32 -> {NUM_CLASSES}")
```

### 15.2 Single Defense Strategy Implementation
```python
from flwr.server.strategy import FedAvg

class SingleDefenseStrategy(FedAvg):
    def __init__(self, defense_strategy, server_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defense_strategy = defense_strategy
        self.detected_anomalies = []
        self.server_data = server_data
        
        # Initialize specific defense
        if defense_strategy == 'KRUM':
            self.defense = KrumDefense(BYZANTINE_CLIENTS)
        elif defense_strategy == 'TRIMMED_MEAN':
            self.defense = TrimmedMeanDefense(TRIM_RATIO)
        elif defense_strategy == 'MEDIAN':
            self.defense = MedianDefense()
        elif defense_strategy == 'ANOMALY_DETECTION':
            self.defense = AnomalyDetectionDefense(ANOMALY_THRESHOLD)
        elif defense_strategy == 'FLTRUST':
            self.defense = FLTrustDefense(FLTRUST_BETA, server_data)
    
    def aggregate_fit(self, server_round, results, failures):
        print(f"\nRound {server_round}: Using {self.defense_strategy} defense")
        
        if self.defense_strategy == 'NONE':
            return super().aggregate_fit(server_round, results, failures)
        
        # Extract client updates
        client_updates = [fl.common.parameters_to_ndarrays(fit_res.parameters) 
                         for _, fit_res in results]
        client_sizes = [fit_res.num_examples for _, fit_res in results]
        
        # Apply anomaly detection defense
        if self.defense_strategy == 'ANOMALY_DETECTION':
            flattened_updates = [np.concatenate([p.flatten() for p in update]) 
                               for update in client_updates]
            anomalous_clients = self.defense.detect_anomalous_clients(flattened_updates)
            
            if anomalous_clients:
                print(f"Detected {len(anomalous_clients)} anomalous clients")
                self.detected_anomalies.append((server_round, anomalous_clients))
                results = [results[i] for i in range(len(results)) if i not in anomalous_clients]
            
            return super().aggregate_fit(server_round, results, failures)
        
        # Apply aggregation-based defenses (Krum, Trimmed Mean, Median, FLTrust)
        else:
            flattened_updates = [np.concatenate([p.flatten() for p in update]) 
                               for update in client_updates]
            
            # For FLTrust, compute server update first
            if self.defense_strategy == 'FLTRUST' and self.server_data is not None:
                # Compute server update using server data
                server_update = self._compute_server_update()
                self.defense.set_server_update(server_update)
            
            # Get defended aggregation
            defended_update = self.defense.aggregate(flattened_updates, client_sizes)
            
            # Reshape back to original parameter structure
            param_shapes = [p.shape for p in client_updates[0]]
            reshaped_params = []
            start_idx = 0
            
            for shape in param_shapes:
                param_size = np.prod(shape)
                param_data = defended_update[start_idx:start_idx + param_size]
                reshaped_params.append(param_data.reshape(shape))
                start_idx += param_size
            
            return fl.common.ndarrays_to_parameters(reshaped_params), {}
    
    def _compute_server_update(self):
        """Compute server update using server data (simplified)"""
        if self.server_data is None:
            return None
        
        # Simplified: return random update as placeholder
        # In practice, this would train on server data
        return np.random.normal(0, 0.1, size=1000)  # Placeholder
    
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        aggregated_accuracy = sum(accuracies) / sum(examples)
        aggregated_loss = sum([r.loss * r.num_examples for _, r in results]) / sum(examples)
        
        print(f"Round {server_round} - Accuracy: {aggregated_accuracy:.4f}")
        
        return aggregated_loss, {"accuracy": aggregated_accuracy}

# Prepare server data for FLTrust (if needed)
server_data = None
if DEFENSE_STRATEGY == 'FLTRUST':
    # Use a small portion of training data as server data
    server_size = int(len(train_df) * SERVER_DATA_RATIO)
    server_data = train_df.sample(n=server_size, random_state=42)
    print(f"Prepared server data: {len(server_data)} samples")

# Initialize strategy
strategy = SingleDefenseStrategy(
    defense_strategy=DEFENSE_STRATEGY,
    server_data=server_data,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_OF_CLIENTS,
    min_evaluate_clients=NUM_OF_CLIENTS,
    min_available_clients=NUM_OF_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
)

print(f"Initialized strategy with {DEFENSE_STRATEGY} defense")
```

## Step 16: Deploy Federated Learning Simulation

### 16.1 Configure Simulation
```python
from flwr.simulation import start_simulation

# Simulation configuration
config = fl.server.ServerConfig(num_rounds=NUM_OF_ROUNDS)

print(f"\n=== Federated Learning Simulation Configuration ===")
print(f"Method: {METHOD}")
print(f"Number of clients: {NUM_OF_CLIENTS}")
print(f"Number of rounds: {NUM_OF_ROUNDS}")
print(f"Malicious clients: {len(malicious_client_ids)} ({malicious_client_ids})")
print(f"Classification type: {class_size}-class ({class_size_map[int(class_size)]})")
print(f"Dataset size: {len(train_df)} training samples")
print(f"Device: {DEVICE}")
print("=" * 50)
```

### 16.2 Run Simulation
```python
# Start federated learning simulation
print("\nStarting Federated Learning Simulation...")

history = start_simulation(
    client_fn=client_fn,
    num_clients=NUM_OF_CLIENTS,
    config=config,
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0.1 if torch.cuda.is_available() else 0},
)

print("\nFederated Learning Simulation Completed!")
```

### 16.3 Results Analysis and Visualization
```python
import matplotlib.pyplot as plt

# Extract metrics from history
round_numbers = list(range(1, NUM_OF_ROUNDS + 1))
accuracies = [history.metrics_distributed["accuracy"][i][1] for i in range(NUM_OF_ROUNDS)]
losses = [history.losses_distributed[i][1] for i in range(NUM_OF_ROUNDS)]

# Plot training progress
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
ax1.plot(round_numbers, accuracies, 'b-o', linewidth=2, markersize=6)
ax1.set_title('Federated Learning Accuracy')
ax1.set_xlabel('Round')
ax1.set_ylabel('Accuracy')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Loss plot
ax2.plot(round_numbers, losses, 'r-o', linewidth=2, markersize=6)
ax2.set_title('Federated Learning Loss')
ax2.set_xlabel('Round')
ax2.set_ylabel('Loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final results
print(f"\n=== Final Results ===")
print(f"Final Accuracy: {accuracies[-1]:.4f}")
print(f"Final Loss: {losses[-1]:.4f}")
print(f"Best Accuracy: {max(accuracies):.4f} (Round {accuracies.index(max(accuracies)) + 1})")
print(f"Attack Impact: {'Significant' if max(accuracies) < 0.7 else 'Limited'}")
```

### 16.4 Model Evaluation on Test Set
```python
# Evaluate final global model on test set
final_model = IoTAttackNet(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
final_parameters = fl.common.parameters_to_ndarrays(history.parameters_distributed[-1][1])
set_model_parameters(final_model, final_parameters)

# Create test dataloader
test_dataset = TensorDataset(
    torch.FloatTensor(test_df[X_columns].values),
    torch.LongTensor(test_df[y_column].values)
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluate
test_loss, test_accuracy = test_model(final_model, test_loader)

print(f"\n=== Test Set Evaluation ===")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate classification report
final_model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = final_model(data)
        pred = output.argmax(dim=1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

# Print classification report
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

print("\n=== Attack Analysis ===")
if len(malicious_client_ids) > 0:
    print(f"Poison attack with {len(malicious_client_ids)} malicious clients")
    print(f"Poison factor: {poison_factor}")
    print(f"Attack success: {'Yes' if test_accuracy < 0.7 else 'No'}")
else:
    print("No poison attack - baseline performance")
```

## Implementation Summary

The complete federated learning implementation includes:

1. **Neural Network**: 4-layer MLP for IoT attack detection
2. **FL Client**: Flower client with poison attack capabilities
3. **FL Server**: FedAvg strategy with optional defense mechanisms
4. **Simulation**: Complete FL training with evaluation and visualization
5. **Security Research**: Model poisoning attacks and impact analysis
6. **Evaluation**: Comprehensive performance metrics and attack assessment

This implementation provides a complete framework for researching federated learning security in IoT attack detection scenarios.

## Step 16.5: Defense Evaluation

### 16.5.1 Single Defense Analysis
```python
# Analyze selected defense strategy
final_accuracy = history.metrics_distributed["accuracy"][-1][1]

print(f"\n=== {DEFENSE_STRATEGY} Defense Results ===")
print(f"Final accuracy: {final_accuracy:.4f}")

if DEFENSE_STRATEGY == 'ANOMALY_DETECTION' and hasattr(strategy, 'detected_anomalies'):
    print(f"Total anomalies detected: {len(strategy.detected_anomalies)}")
    for round_num, anomalies in strategy.detected_anomalies:
        print(f"Round {round_num}: {len(anomalies)} anomalous clients")
elif DEFENSE_STRATEGY == 'FLTRUST':
    print(f"Applied FLTrust defense with beta={FLTRUST_BETA}")
elif DEFENSE_STRATEGY != 'NONE':
    print(f"Applied {DEFENSE_STRATEGY} aggregation defense")
else:
    print("No defense applied (baseline)")
```

### 16.5.2 Quick Defense Comparison
```python
def test_single_defense(defense_name):
    """Test a single defense strategy"""
    print(f"\nTesting {defense_name} defense...")
    
    # This would require re-running the simulation with different DEFENSE_STRATEGY
    # For demonstration purposes, showing the structure
    
    # DEFENSE_STRATEGY = defense_name
    # strategy = SingleDefenseStrategy(defense_name, ...)
    # history = start_simulation(...)
    # return history.metrics_distributed["accuracy"][-1][1]
    
    return f"Would test {defense_name} defense here"

# Example usage:
# krum_accuracy = test_single_defense('KRUM')
# trimmed_accuracy = test_single_defense('TRIMMED_MEAN')
# median_accuracy = test_single_defense('MEDIAN')
# anomaly_accuracy = test_single_defense('ANOMALY_DETECTION')
```
