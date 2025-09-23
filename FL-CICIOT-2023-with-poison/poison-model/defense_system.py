"""
Federated Learning Defense System
FLTrust and Krum Defense Implementations with Configuration
"""

import numpy as np
from scipy.spatial.distance import cosine

# Defense Configuration
DEFENSE_STRATEGY = 'FLTRUST'  # Options: 'NONE', 'KRUM', 'FLTRUST'

FLTRUST_CONFIG = {
    'server_data_ratio': 0.1,
    'beta': 0.5
}

KRUM_CONFIG = {
    'num_byzantine': 2,
    'k': 3
}

def get_defense_config(strategy):
    """Get configuration for defense strategy"""
    configs = {
        'FLTRUST': FLTRUST_CONFIG,
        'KRUM': KRUM_CONFIG
    }
    return configs.get(strategy.upper(), {})

# Defense Implementations
class FLTrustDefense:
    def __init__(self, server_data_ratio=0.1, beta=0.5):
        self.server_data_ratio = server_data_ratio
        self.beta = beta
        self.server_data = None
        
    def prepare_server_data(self, train_df, X_columns, y_column):
        server_size = int(len(train_df) * self.server_data_ratio)
        self.server_data = train_df.sample(n=server_size, random_state=42)
        return self.server_data
        
    def compute_trust_scores(self, client_updates, server_update):
        trust_scores = []
        server_flat = np.concatenate([p.flatten() for p in server_update])
        
        for update in client_updates:
            client_flat = np.concatenate([p.flatten() for p in update])
            similarity = np.dot(server_flat, client_flat) / (
                np.linalg.norm(server_flat) * np.linalg.norm(client_flat) + 1e-8
            )
            trust_scores.append(max(0, similarity))
            
        return np.array(trust_scores)
        
    def aggregate(self, client_updates, server_update):
        trust_scores = self.compute_trust_scores(client_updates, server_update)
        
        if trust_scores.sum() > 0:
            trust_scores = trust_scores / trust_scores.sum()
        else:
            trust_scores = np.ones(len(client_updates)) / len(client_updates)
            
        aggregated = [np.zeros_like(param) for param in client_updates[0]]
        for i, update in enumerate(client_updates):
            for j, param in enumerate(update):
                aggregated[j] += trust_scores[i] * param
                
        return aggregated, trust_scores

class KrumDefense:
    def __init__(self, num_byzantine=2):
        self.num_byzantine = num_byzantine
        
    def compute_distances(self, client_updates):
        n_clients = len(client_updates)
        distances = np.zeros((n_clients, n_clients))
        
        flattened_updates = []
        for update in client_updates:
            flattened = np.concatenate([p.flatten() for p in update])
            flattened_updates.append(flattened)
        
        for i in range(n_clients):
            for j in range(i+1, n_clients):
                dist = np.linalg.norm(flattened_updates[i] - flattened_updates[j])
                distances[i, j] = distances[j, i] = dist
                
        return distances
        
    def aggregate(self, client_updates):
        distances = self.compute_distances(client_updates)
        n_clients = distances.shape[0]
        scores = []
        
        for i in range(n_clients):
            client_distances = distances[i]
            sorted_distances = np.sort(client_distances)[1:n_clients-self.num_byzantine-1]
            scores.append(np.sum(sorted_distances))
            
        selected_client = np.argmin(scores)
        return client_updates[selected_client], selected_client

def create_defense(defense_type, **kwargs):
    """Factory function to create defense instances"""
    if defense_type.upper() == 'FLTRUST':
        return FLTrustDefense(
            server_data_ratio=kwargs.get('server_data_ratio', 0.1),
            beta=kwargs.get('beta', 0.5)
        )
    elif defense_type.upper() == 'KRUM':
        return KrumDefense(
            num_byzantine=kwargs.get('num_byzantine', 2)
        )
    else:
        return None