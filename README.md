# Privacy-Preserving Federated Learning

[![Python Requirements](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flower](https://img.shields.io/badge/framework-flower-ff69b4.svg)](https://flower.dev/)
[![Privacy: Opacus](https://img.shields.io/badge/privacy-opacus-green.svg)](https://opacus.ai/)
[![ML: PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)

A modernized, privacy-first decentralized Machine Learning pipeline that trains a Convolutional Neural Network (CNN) across distributed edge clients on the CIFAR-10 dataset without centralizing their raw data. 

This repository leverages mathematical noise injection (Differential Privacy) and federated aggregation to build performance-conscious models while strictly maintaining data sovereignty.

---

## 🌟 Key Features

* **Data Sovereignty via Federated Learning (FedAvg):** Powered by the **Flower** framework, this system ensures that client data never leaves the local device. The server only aggregates encrypted or noise-infused model weight updates.
* **Differential Privacy (DP):** Integrated via **Opacus**, mathematical noise is dynamically scaled and injected into model parameters at the edge layer, guarding against model inversion and Membership Inference Attacks (MIA).
* **Robust Software Architecture:** Engineered with professional grade static-typing, comprehensive modularity (`src/task.py`, decoupled network configurations), and cross-platform hardware support allowing dynamic device allocation (CPU, CUDA, MPS).

## 🛠️ Tech Stack & Architecture

- **PyTorch** for constructing and training the Convolutional Neural Network (CNN).
- **Flower (flwr)** for managing the orchestration of communication rounds between the central server and virtual clients.
- **Opacus** for calculating privacy budgets per epoch and dynamically scaling the clipping and noise functions.
- **Torchvision** for streamlined ingestion and transformation of the CIFAR-10 dataset.

## 🚀 Quick Start

### 1. Installation

Ensure you have a modern version of Python (3.9+) installed. Create a virtual environment and load the dependencies:

```bash
# Create mapping environment
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Simulation

The project includes a robust native runloop that bypasses cumbersome Ray engine setups to provide a low-overhead simulation environment. 

To kick off the Federated Learning simulation across all clients:

```bash
python run_simulation.py
```

## 📊 Analytics and Epsilon Tracking

During execution, the orchestrator logs robust metrics, providing transparency into model performance vs. privacy budget depletion.

**Sample Log Feed:**
```text
======== ROUND 3 ========
-> Sending global weights to Client 0...
   [Client 0] Complete. DP Privacy Budget (Epsilon): 0.4810
-> Sending global weights to Client 1...
   [Client 1] Complete. DP Privacy Budget (Epsilon): 0.4810
...
<- Aggregating Model Updates on Server (FedAvg)
<- Evaluating Updated Global Model...
======== ROUND 3 SUMMARY ========
Global Loss: 2.3067 | Global Accuracy: 10.30%
```

Even with strict differential privacy enabled (e.g., Epsilon \(\approx 0.48\)), the framework establishes verifiable client-server synchronization, laying the foundation to perform hyper-parameter tuning (like exploring noise multipliers or client subsetting) depending on your target security profile.

---

*Authored for security engineering portfolios and privacy-focused ML researchers. Contributions are welcome!*
