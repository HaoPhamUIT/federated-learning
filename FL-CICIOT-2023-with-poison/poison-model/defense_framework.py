# Generic Defense Framework for Model Poisoning Attacks (MPA)
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
import flwr as fl
from flwr.common import Parameters

# Base Defense Class
class BaseDefense(ABC):
    def __init__(self, model, root_dataset=None, **kwargs):
        self.model = model
        self.root_dataset = root_dataset
        self.config = kwargs
    
    @abstractmethod
    def aggregate(self, client_updates, global_params):
        """Aggregate client updates with defense mechanism"""
        pass

# FLTrust Defense
class FLTrustDefense(BaseDefense):
    def __init__(self, model, root_dataset, trust_threshold=0.1):
        super().__init__(model, root_dataset, trust_threshold=trust_threshold)
        self.trust_threshold = trust_threshold
    
    def compute_server_gradient(self, global_params):
        self.model.load_state_dict(global_params)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(self.root_dataset):
            if batch_idx >= 5: break
            output = self.model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
        
        return torch.cat([p.grad.clone().flatten() for p in self.model.parameters() if p.grad is not None])
    
    def aggregate(self, client_updates, global_params):
        server_grad = self.compute_server_gradient(global_params)
        trust_scores = []
        
        for client_params in client_updates:
            client_state = fl.common.parameters_to_ndarrays(client_params)
            client_grad = torch.cat([torch.tensor(g - c).flatten() 
                                   for g, c in zip(global_params.values(), client_state)])
            cos_sim = torch.nn.functional.cosine_similarity(
                server_grad.unsqueeze(0), client_grad.unsqueeze(0)).item()
            trust_scores.append(max(0, cos_sim))
        
        # Filter and aggregate
        filtered_updates = [u for u, s in zip(client_updates, trust_scores) if s >= self.trust_threshold]
        filtered_scores = [s for s in trust_scores if s >= self.trust_threshold]
        
        if not filtered_updates:
            filtered_updates, filtered_scores = client_updates, trust_scores
        
        weights = [s/sum(filtered_scores) for s in filtered_scores] if sum(filtered_scores) > 0 else [1/len(filtered_scores)] * len(filtered_scores)
        client_arrays = [fl.common.parameters_to_ndarrays(p) for p in filtered_updates]
        
        aggregated = [sum(w * c[i] for w, c in zip(weights, client_arrays)) for i in range(len(client_arrays[0]))]
        return fl.common.ndarrays_to_parameters(aggregated), trust_scores

# Krum Defense
class KrumDefense(BaseDefense):
    def __init__(self, model, root_dataset=None, f=2):
        super().__init__(model, root_dataset, f=f)
        self.f = f  # number of Byzantine clients
    
    def aggregate(self, client_updates, global_params):
        client_arrays = [fl.common.parameters_to_ndarrays(p) for p in client_updates]
        n = len(client_arrays)
        
        # Compute pairwise distances
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = sum(np.linalg.norm(client_arrays[i][k] - client_arrays[j][k]) 
                             for k in range(len(client_arrays[i])))
                    distances.append(dist)
            distances.sort()
            scores.append(sum(distances[:n-self.f-2]))
        
        # Select client with minimum score
        selected_idx = np.argmin(scores)
        return client_updates[selected_idx], [1 if i == selected_idx else 0 for i in range(n)]

# Median Defense
class MedianDefense(BaseDefense):
    def aggregate(self, client_updates, global_params):
        client_arrays = [fl.common.parameters_to_ndarrays(p) for p in client_updates]
        aggregated = [np.median([c[i] for c in client_arrays], axis=0) for i in range(len(client_arrays[0]))]
        return fl.common.ndarrays_to_parameters(aggregated), [1] * len(client_updates)

# Trimmed Mean Defense
class TrimmedMeanDefense(BaseDefense):
    def __init__(self, model, root_dataset=None, trim_ratio=0.2):
        super().__init__(model, root_dataset, trim_ratio=trim_ratio)
        self.trim_ratio = trim_ratio
    
    def aggregate(self, client_updates, global_params):
        client_arrays = [fl.common.parameters_to_ndarrays(p) for p in client_updates]
        n = len(client_arrays)
        trim_count = int(n * self.trim_ratio)
        
        aggregated = []
        for i in range(len(client_arrays[0])):
            layer_params = np.array([c[i] for c in client_arrays])
            sorted_params = np.sort(layer_params, axis=0)
            trimmed = sorted_params[trim_count:n-trim_count]
            aggregated.append(np.mean(trimmed, axis=0))
        
        return fl.common.ndarrays_to_parameters(aggregated), [1] * len(client_updates)

# Attack Client Classes
class MimicAttackClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, target_updates=None, attack_intensity=0.7):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.target_updates = target_updates
        self.attack_intensity = attack_intensity
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # Minimal training
        for batch_idx, (data, target) in enumerate(self.trainloader):
            if batch_idx >= 3: break
            optimizer.zero_grad()
            output = self.model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
        
        # Apply attack
        if self.target_updates is not None:
            current_params = self.get_parameters({})
            for i, (current, target) in enumerate(zip(current_params, self.target_updates)):
                current_params[i] = (1 - self.attack_intensity) * current + self.attack_intensity * target
            self.set_parameters(current_params)
        
        return self.get_parameters({}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in self.valloader:
                output = self.model(data)
                loss += torch.nn.CrossEntropyLoss()(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return loss, len(self.valloader.dataset), {"accuracy": correct / len(self.valloader.dataset)}

# Benign Client
class BenignClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        for batch_idx, (data, target) in enumerate(self.trainloader):
            if batch_idx >= 10: break
            optimizer.zero_grad()
            output = self.model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
        
        return self.get_parameters({}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        return self.model.evaluate(parameters, config)

# IoT Neural Network
class IoTNet(torch.nn.Module):
    def __init__(self, input_dim=39, num_classes=34):
        super(IoTNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.dropout = torch.nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Generic Experiment Runner
def run_defense_experiment(defense_class, defense_params=None, num_clients=8, num_attackers=2, rounds=5):
    """Generic function to run any defense against MPA"""
    if defense_params is None:
        defense_params = {}
    
    # Use existing data
    X = train_df.drop('Label', axis=1).values.astype(np.float32)
    y = train_df['Label'].values.astype(np.int64)
    
    # Create client data
    samples_per_client = len(X) // num_clients
    client_data = []
    for i in range(num_clients):
        start, end = i * samples_per_client, (i + 1) * samples_per_client
        client_X, client_y = torch.tensor(X[start:end]), torch.tensor(y[start:end])
        split = int(0.8 * len(client_X))
        train_dataset = torch.utils.data.TensorDataset(client_X[:split], client_y[:split])
        val_dataset = torch.utils.data.TensorDataset(client_X[split:], client_y[split:])
        client_data.append((torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True),
                           torch.utils.data.DataLoader(val_dataset, batch_size=64)))
    
    # Server data (for defenses that need it)
    server_data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X[:1000]), torch.tensor(y[:1000])), batch_size=32)
    
    # Initialize defense
    num_classes = len(np.unique(y))
    global_model = IoTNet(input_dim=X.shape[1], num_classes=num_classes)
    defense = defense_class(global_model, server_data, **defense_params)
    
    print(f"Running {defense_class.__name__} with {num_clients} clients ({num_attackers} attackers)...")
    
    # FL simulation
    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1} ---")
        client_updates = []
        global_params = global_model.state_dict()
        
        for client_id in range(num_clients):
            train_loader, val_loader = client_data[client_id]
            
            if client_id >= num_clients - num_attackers:
                # Attack client
                target_updates = [np.random.normal(0, 0.01, size=p.shape) for p in global_model.parameters()]
                client = MimicAttackClient(IoTNet(input_dim=X.shape[1], num_classes=num_classes),
                                         train_loader, val_loader, target_updates)
            else:
                # Benign client
                client = BenignClient(IoTNet(input_dim=X.shape[1], num_classes=num_classes),
                                    train_loader, val_loader)
            
            # Train client
            initial_params = [val.cpu().numpy() for _, val in global_params.items()]
            client.set_parameters(initial_params)
            updated_params, _, _ = client.fit(initial_params, {})
            client_updates.append(fl.common.ndarrays_to_parameters(updated_params))
        
        # Apply defense
        aggregated_params, scores = defense.aggregate(client_updates, global_params)
        global_model.load_state_dict({k: torch.tensor(v) for k, v in 
                                    zip(global_model.state_dict().keys(), 
                                        fl.common.parameters_to_ndarrays(aggregated_params))})
        
        print(f"Defense scores: {[f'{s:.3f}' for s in scores]}")
    
    print(f"\n{defense_class.__name__} simulation completed!")
    return global_model, scores