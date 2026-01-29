#!/usr/bin/env python3
"""Example script demonstrating domain generalization usage."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from utils import set_seed, get_device
from models import DomainGeneralizationModel
from data import DomainGeneralizationDataModule, SyntheticDomainDataset, DomainDataset
from train import DomainGeneralizationTrainer
from eval import DomainGeneralizationEvaluator
import logging

def main():
    """Run a simple domain generalization example."""
    print("Domain Generalization Example")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create synthetic datasets for demonstration
    print("Creating synthetic datasets...")
    source_dataset = SyntheticDomainDataset(
        num_samples=200,
        num_classes=5,
        image_size=64,
        domain_style="cartoon"
    )
    
    target_dataset = SyntheticDomainDataset(
        num_samples=100,
        num_classes=5,
        image_size=64,
        domain_style="sketch"
    )
    
    # Create domain dataset
    domain_dataset = DomainDataset(
        [source_dataset, target_dataset],
        [0, 1]  # Domain labels
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        domain_dataset, batch_size=16, shuffle=True
    )
    
    # Create model
    print("Initializing model...")
    model = DomainGeneralizationModel(
        method="dann",
        num_classes=5,
        backbone="resnet18",
        pretrained=False,  # Use False for synthetic data
        feature_dim=256
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    config = {
        "max_epochs": 3,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": None,
        "patience": 10,
        "min_delta": 0.001,
        "alpha": 1.0
    }
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    trainer = DomainGeneralizationTrainer(
        model, None, config, device, logger
    )
    
    # Train for a few epochs
    print("Training model...")
    for epoch in range(3):
        train_metrics = trainer.train_epoch(data_loader)
        print(f"Epoch {epoch + 1}: Loss = {train_metrics['loss']:.4f}, "
              f"Accuracy = {train_metrics['accuracy']:.2f}%")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        images, class_labels, domain_labels = next(iter(data_loader))
        images = images.to(device)
        
        # Forward pass
        if model.method == "dann":
            class_output, domain_output = model(images)
            
            # Get predictions
            class_pred = torch.argmax(class_output, dim=1)
            domain_pred = torch.argmax(domain_output, dim=1)
            
            print(f"Class predictions: {class_pred.cpu().numpy()}")
            print(f"Domain predictions: {domain_pred.cpu().numpy()}")
            print(f"True classes: {class_labels.numpy()}")
            print(f"True domains: {domain_labels.numpy()}")
            
        else:
            class_output = model(images)
            class_pred = torch.argmax(class_output, dim=1)
            
            print(f"Class predictions: {class_pred.cpu().numpy()}")
            print(f"True classes: {class_labels.numpy()}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
