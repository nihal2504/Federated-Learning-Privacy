import sys
import os
import torch
from collections import OrderedDict

# Ensure our src package is loadable
sys.path.append(os.path.abspath('./src'))

# Import our optimized functions
from task import load_data, Net, FlowerClient, get_device
from flwr.server.strategy.aggregate import aggregate

def log(msg="", log_file="simulation_output.md"):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_local_simulation(num_clients=10, num_rounds=3):
    device = get_device()
    
    # Initialize clean output file
    with open("simulation_output.md", "w", encoding="utf-8") as f:
        f.write("# Federated Learning Differential Privacy Simulation\n\n```text\n")
        
    log(f"\n🚀 Starting Native Federated Learning Simulation with DP on: {device}\n")
    log("Bypassing Ray Engine to simulate FedAvg natively...\n")
    
    global_model = Net().to(device)
    global_weights = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    
    for round_num in range(1, num_rounds + 1):
        log(f"\n======== ROUND {round_num} ========")
        client_results = []
        
        for client_id in range(num_clients):
            log(f"-> Sending global weights to Client {client_id}...")
            trainloader, valloader = load_data(client_id, batch_size=32)
            
            # Spin up client
            client_net = Net().to(device)
            client = FlowerClient(client_net, trainloader, valloader, device)
            
            # Train and inject DP noise
            updated_weights, num_examples, metrics = client.fit(global_weights, {})
            eps = metrics.get('epsilon', 0.0)
            log(f"   [Client {client_id}] Complete. DP Privacy Budget (Epsilon): {eps:.4f}")
            
            client_results.append((updated_weights, num_examples))
            
        log("\n<- Aggregating Model Updates on Server (FedAvg)")
        # Native Flower Federated Averaging algorithm
        global_weights = aggregate(client_results)
        
        log("<- Evaluating Updated Global Model...")
        # Evaluate global model on a single validation set for performance tracking
        _, valloader = load_data(0, batch_size=32)
        eval_net = Net().to(device)
        eval_client = FlowerClient(eval_net, None, valloader, device)
        loss, _, eval_metrics = eval_client.evaluate(global_weights, {})
        acc = eval_metrics.get('accuracy', 0.0)
        
        log(f"======== ROUND {round_num} SUMMARY ========")
        log(f"Global Loss: {loss:.4f} | Global Accuracy: {acc*100:.2f}%\n")
        
    # Close off the Markdown code block
    with open("simulation_output.md", "a", encoding="utf-8") as f:
        f.write("```\n")

if __name__ == "__main__":
    run_local_simulation()

