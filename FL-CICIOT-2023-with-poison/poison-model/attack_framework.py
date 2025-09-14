# Generic Model Poisoning Attacks Framework
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
import flwr as fl

# Base Attack Class
class BaseAttack(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    def poison_update(self, model, parameters, trainloader, **kwargs):
        """Apply attack to model parameters"""
        pass

# Mimic Attack
class MimicAttack(BaseAttack):
    def __init__(self, attack_intensity=0.7, target_updates=None):
        super().__init__(attack_intensity=attack_intensity, target_updates=target_updates)
        self.attack_intensity = attack_intensity
        self.target_updates = target_updates
    
    def poison_update(self, model, parameters, trainloader, **kwargs):
        # Normal training first
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for batch_idx, (data, target) in enumerate(trainloader):
            if batch_idx >= 3: break
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
        
        # Apply mimic attack
        if self.target_updates is not None:
            current_params = [val.cpu().numpy() for _, val in model.state_dict().items()]
            for i, (current, target) in enumerate(zip(current_params, self.target_updates)):
                current_params[i] = (1 - self.attack_intensity) * current + self.attack_intensity * target
            
            # Load poisoned parameters
            params_dict = zip(model.state_dict().keys(), current_params)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
        
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Label Flipping Attack
class LabelFlipAttack(BaseAttack):
    def __init__(self, flip_ratio=0.3, target_label=0):
        super().__init__(flip_ratio=flip_ratio, target_label=target_label)
        self.flip_ratio = flip_ratio
        self.target_label = target_label
    
    def poison_update(self, model, parameters, trainloader, **kwargs):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for batch_idx, (data, target) in enumerate(trainloader):
            if batch_idx >= 10: break
            
            # Flip labels randomly
            flip_mask = torch.rand(target.size()) < self.flip_ratio
            target[flip_mask] = self.target_label
            
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
        
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Backdoor Attack
class BackdoorAttack(BaseAttack):
    def __init__(self, trigger_pattern=None, target_label=0, poison_ratio=0.1):
        super().__init__(trigger_pattern=trigger_pattern, target_label=target_label, poison_ratio=poison_ratio)
        self.trigger_pattern = trigger_pattern or torch.ones(5)  # Simple trigger
        self.target_label = target_label
        self.poison_ratio = poison_ratio
    
    def poison_update(self, model, parameters, trainloader, **kwargs):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for batch_idx, (data, target) in enumerate(trainloader):
            if batch_idx >= 10: break
            
            # Add backdoor trigger to some samples
            poison_mask = torch.rand(data.size(0)) < self.poison_ratio
            if poison_mask.any():
                # Add trigger pattern to first few features
                trigger_size = min(len(self.trigger_pattern), data.size(1))
                data[poison_mask, :trigger_size] = self.trigger_pattern[:trigger_size]
                target[poison_mask] = self.target_label
            
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
        
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Byzantine Attack (Random Noise)
class ByzantineAttack(BaseAttack):
    def __init__(self, noise_scale=1.0):
        super().__init__(noise_scale=noise_scale)
        self.noise_scale = noise_scale
    
    def poison_update(self, model, parameters, trainloader, **kwargs):
        # Minimal training
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for batch_idx, (data, target) in enumerate(trainloader):
            if batch_idx >= 2: break
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
        
        # Add random noise to parameters
        poisoned_params = []
        for _, param in model.state_dict().items():
            noise = torch.randn_like(param) * self.noise_scale
            poisoned_param = param + noise
            poisoned_params.append(poisoned_param.cpu().numpy())
        
        return poisoned_params

# Gradient Ascent Attack
class GradientAscentAttack(BaseAttack):
    def __init__(self, ascent_steps=5, lr=0.1):
        super().__init__(ascent_steps=ascent_steps, lr=lr)
        self.ascent_steps = ascent_steps
        self.lr = lr
    
    def poison_update(self, model, parameters, trainloader, **kwargs):
        model.train()
        
        # Gradient ascent to maximize loss
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        for step in range(self.ascent_steps):
            for batch_idx, (data, target) in enumerate(trainloader):
                if batch_idx >= 3: break
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.CrossEntropyLoss()(output, target)
                # Gradient ascent (maximize loss)
                (-loss).backward()
                optimizer.step()
        
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Model Replacement Attack
class ModelReplacementAttack(BaseAttack):
    def __init__(self, replacement_model=None):
        super().__init__(replacement_model=replacement_model)
        self.replacement_model = replacement_model
    
    def poison_update(self, model, parameters, trainloader, **kwargs):
        if self.replacement_model is not None:
            # Replace with pre-trained malicious model
            return [val.cpu().numpy() for _, val in self.replacement_model.state_dict().items()]
        else:
            # Random model replacement
            for param in model.parameters():
                param.data = torch.randn_like(param.data)
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Generic Attack Client
class AttackClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, attack_strategy):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.attack = attack_strategy
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Apply attack
        poisoned_params = self.attack.poison_update(
            self.model, parameters, self.trainloader
        )
        
        return poisoned_params, len(self.trainloader.dataset), {}
    
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

# Attack Factory
class AttackFactory:
    @staticmethod
    def create_attack(attack_type, **kwargs):
        attacks = {
            'mimic': MimicAttack,
            'label_flip': LabelFlipAttack,
            'backdoor': BackdoorAttack,
            'byzantine': ByzantineAttack,
            'gradient_ascent': GradientAscentAttack,
            'model_replacement': ModelReplacementAttack
        }
        
        if attack_type not in attacks:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return attacks[attack_type](**kwargs)

# Generic Experiment with Attacks
def run_attack_experiment(defense_class, attack_configs, defense_params=None, num_clients=8, rounds=5):
    """Run experiment with multiple attack types"""
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
    
    # Server data
    server_data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X[:1000]), torch.tensor(y[:1000])), batch_size=32)
    
    # Initialize defense
    num_classes = len(np.unique(y))
    global_model = IoTNet(input_dim=X.shape[1], num_classes=num_classes)
    defense = defense_class(global_model, server_data, **defense_params)
    
    print(f"Running {defense_class.__name__} against {len(attack_configs)} attack types...")
    
    # FL simulation
    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1} ---")
        client_updates = []
        global_params = global_model.state_dict()
        
        for client_id in range(num_clients):
            train_loader, val_loader = client_data[client_id]
            
            # Create client based on attack config
            if client_id < len(attack_configs):
                attack_config = attack_configs[client_id]
                attack = AttackFactory.create_attack(**attack_config)
                client = AttackClient(IoTNet(input_dim=X.shape[1], num_classes=num_classes),
                                    train_loader, val_loader, attack)
                print(f"Client {client_id}: {attack_config['attack_type']} attack")
            else:
                # Benign client
                from defense_framework import BenignClient
                client = BenignClient(IoTNet(input_dim=X.shape[1], num_classes=num_classes),
                                    train_loader, val_loader)
                print(f"Client {client_id}: Benign")
            
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
    
    print(f"\n{defense_class.__name__} vs attacks simulation completed!")
    return global_model, scores