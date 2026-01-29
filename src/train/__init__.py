"""Training framework for domain generalization."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
import json
import time

from ..utils import EarlyStopping, save_checkpoint, load_checkpoint
from ..models import DomainGeneralizationModel
from .data import DomainGeneralizationDataModule


class DomainGeneralizationTrainer:
    """Trainer for domain generalization models."""
    
    def __init__(
        self,
        model: DomainGeneralizationModel,
        data_module: DomainGeneralizationDataModule,
        config: Dict[str, Any],
        device: torch.device,
        logger: logging.Logger
    ):
        """Initialize trainer.
        
        Args:
            model: Domain generalization model
            data_module: Data module
            config: Training configuration
            device: Device to train on
            logger: Logger instance
        """
        self.model = model.to(device)
        self.data_module = data_module
        self.config = config
        self.device = device
        self.logger = logger
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss functions
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 0.001)
        )
        
        # Gradient reversal parameters
        self.alpha = config.get("alpha", 1.0)
        self.alpha_scheduler = self._setup_alpha_scheduler()
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", None)
        
        if scheduler_name is None:
            return None
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("step_size", 30),
                gamma=self.config.get("gamma", 0.1)
            )
        elif scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("max_epochs", 100)
            )
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _setup_alpha_scheduler(self) -> Optional[Any]:
        """Setup alpha scheduler for gradient reversal."""
        if self.config.get("method") == "dann":
            return lambda epoch: 2.0 / (1.0 + np.exp(-10 * epoch / self.config.get("max_epochs", 100))) - 1.0
        return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_domain_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, class_labels, domain_labels) in enumerate(progress_bar):
            images = images.to(self.device)
            class_labels = class_labels.to(self.device)
            domain_labels = domain_labels.to(self.device)
            
            # Update alpha for gradient reversal
            if self.alpha_scheduler is not None:
                current_alpha = self.alpha_scheduler(self.current_epoch)
            else:
                current_alpha = self.alpha
            
            # Forward pass
            if self.config.get("method") == "dann":
                outputs = self.model(images, alpha=current_alpha)
                losses = self.model.get_loss(
                    outputs,
                    class_labels,
                    domain_labels=domain_labels
                )
            else:
                outputs = self.model(images)
                losses = self.model.get_loss(outputs, class_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses["total"].item()
            total_classification_loss += losses["classification"].item()
            
            if "domain" in losses:
                total_domain_loss += losses["domain"].item()
            
            # Calculate accuracy
            if isinstance(outputs, tuple):
                class_output = outputs[0]
            else:
                class_output = outputs
            
            _, predicted = torch.max(class_output.data, 1)
            total += class_labels.size(0)
            correct += (predicted == class_labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{losses['total'].item():.4f}",
                "Acc": f"{100 * correct / total:.2f}%"
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_classification_loss = total_classification_loss / len(train_loader)
        epoch_domain_loss = total_domain_loss / len(train_loader) if total_domain_loss > 0 else 0.0
        epoch_acc = 100 * correct / total
        
        return {
            "loss": epoch_loss,
            "classification_loss": epoch_classification_loss,
            "domain_loss": epoch_domain_loss,
            "accuracy": epoch_acc
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, class_labels, domain_labels in val_loader:
                images = images.to(self.device)
                class_labels = class_labels.to(self.device)
                
                # Forward pass
                if self.config.get("method") == "dann":
                    outputs = self.model(images, alpha=0.0)  # No gradient reversal during validation
                    losses = self.model.get_loss(outputs, class_labels)
                else:
                    outputs = self.model(images)
                    losses = self.model.get_loss(outputs, class_labels)
                
                total_loss += losses["total"].item()
                
                # Calculate accuracy
                if isinstance(outputs, tuple):
                    class_output = outputs[0]
                else:
                    class_output = outputs
                
                _, predicted = torch.max(class_output.data, 1)
                total += class_labels.size(0)
                correct += (predicted == class_labels).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc
        }
    
    def test_epoch(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test for one epoch.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of test metrics
        """
        return self.validate_epoch(test_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        save_dir: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            save_dir: Directory to save checkpoints
            
        Returns:
            Dictionary of training history
        """
        max_epochs = self.config.get("max_epochs", 100)
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["accuracy"])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{max_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Store metrics
            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_metrics["loss"])
            self.val_accuracies.append(val_metrics["accuracy"])
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                })
            
            # Save checkpoint
            if save_dir is not None:
                is_best = val_metrics["accuracy"] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics["accuracy"]
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics["loss"],
                    val_metrics,
                    save_dir / "checkpoint.pth",
                    is_best=is_best
                )
            
            # Early stopping
            if self.early_stopping(val_metrics["loss"], self.model):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Final test
        if test_loader is not None:
            test_metrics = self.test_epoch(test_loader)
            self.logger.info(f"Final test accuracy: {test_metrics['accuracy']:.2f}%")
            
            if wandb.run is not None:
                wandb.log({"test/accuracy": test_metrics["accuracy"]})
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies
        }
    
    def evaluate_domain_generalization(
        self,
        source_loaders: Dict[str, DataLoader],
        target_loaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate domain generalization performance.
        
        Args:
            source_loaders: Source domain data loaders
            target_loaders: Target domain data loaders
            
        Returns:
            Dictionary of evaluation results
        """
        self.model.eval()
        results = {}
        
        # Evaluate on source domains
        for domain_name, loader in source_loaders.items():
            metrics = self.validate_epoch(loader)
            results[f"source_{domain_name}"] = metrics
        
        # Evaluate on target domains
        for domain_name, loader in target_loaders.items():
            metrics = self.validate_epoch(loader)
            results[f"target_{domain_name}"] = metrics
        
        return results
