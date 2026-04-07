import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets import FederatedDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import flwr as fl
from collections import OrderedDict
from typing import Tuple, Dict, Any

# Ensure device agnostic execution (CUDA, MPS, or CPU)
def get_device() -> torch.device:
    """Returns the most optimal device available: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class Net(nn.Module):
    """A standard Convolutional Neural Network for CIFAR-10."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def load_data(partition_id: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Loads and partitions the CIFAR-10 dataset."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    partition = fds.load_partition(partition_id, "train")
    partition_train_test = partition.train_test_split(test_size=0.2)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, valloader

def train(
    net: nn.Module, 
    trainloader: DataLoader, 
    epochs: int, 
    device: torch.device,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    lr: float = 0.001
) -> float:
    """Trains the model using Differential Privacy and returns the privacy budget (epsilon)."""
    # Validate the model for Differential Privacy compliance
    net = ModuleValidator.fix(net)
    ModuleValidator.validate(net, strict=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    privacy_engine = PrivacyEngine()
    
    net, optimizer, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            criterion(net(images), labels).backward()
            optimizer.step()
    
    # Calculate and return epsilon
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    return epsilon

def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluates the model on the validation dataset."""
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(testloader), correct / total

class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient optimized for Differential Privacy training."""
    def __init__(self, net: nn.Module, trainloader: DataLoader, valloader: DataLoader, device: torch.device):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def get_parameters(self, config: Dict[str, Any]) -> list:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: list):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list, config: Dict[str, Any]) -> Tuple[list, int, Dict[str, Any]]:
        self.set_parameters(parameters)
        # We explicitly track and output the DP budget after training
        epsilon = train(self.net, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"epsilon": epsilon}

    def evaluate(self, parameters: list, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}

