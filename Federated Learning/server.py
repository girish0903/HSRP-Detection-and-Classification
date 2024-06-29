import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from model import create_model
import os
import tensorflow as tf  # Import TensorFlow module

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate accuracy using a weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def save_model_weights(model: tf.keras.Model, save_path: str):
    """Save the weights of the trained model."""
    model.save_weights(save_path)

def main():
    # Create the model
    model = create_model()

    # Start Flower server with custom strategy
    strategy = fl.server.strategy.FedAvg(
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # Save model weights after training
    save_model_weights(model, "federated_model_weights.weights.h5")

if __name__ == "__main__":
    main()
