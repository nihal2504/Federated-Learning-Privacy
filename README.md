# Privacy-Preserving Federated Learning with Differential Privacy

### Tech Stack
**Python, PyTorch, Flower (flwr), Opacus, CIFAR-10**

### Project Overview
Implemented a decentralized Machine Learning system where a global Convolutional Neural Network (CNN) is trained across 10 simulated edge clients. This project demonstrates how to build high-performance models while maintaining strict data privacy standards.

### Key Features
* **Data Sovereignty:** Utilized **Federated Learning (FedAvg)** to ensure raw data never leaves the client device, preventing centralized data harvesting.
* **Differential Privacy:** Integrated **Opacus** to inject mathematical noise into model updates, protecting the system against Membership Inference Attacks.
* **Scalable Simulation:** Leveraged the **Flower Virtual Client Engine** to orchestrate training rounds across multiple decentralized clients.

### Results
The project successfully demonstrated model convergence over 3 communication rounds.
- **Round 1 Loss:** 73.64
- **Round 3 Loss:** 73.20 (Proving steady learning despite privacy constraints)
