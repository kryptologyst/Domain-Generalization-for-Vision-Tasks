"""Evaluation metrics and leaderboard for domain generalization."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
import json
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ..models import DomainGeneralizationModel
from ..utils import get_device


class DomainGeneralizationEvaluator:
    """Evaluator for domain generalization models."""
    
    def __init__(
        self,
        model: DomainGeneralizationModel,
        device: torch.device,
        logger: logging.Logger
    ):
        """Initialize evaluator.
        
        Args:
            model: Domain generalization model
            device: Device to evaluate on
            logger: Logger instance
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.model.eval()
    
    def evaluate_classification(
        self,
        data_loader: torch.utils.data.DataLoader,
        domain_name: str = "unknown"
    ) -> Dict[str, float]:
        """Evaluate classification performance.
        
        Args:
            data_loader: Data loader
            domain_name: Name of the domain
            
        Returns:
            Dictionary of metrics
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, class_labels, domain_labels in data_loader:
                images = images.to(self.device)
                class_labels = class_labels.to(self.device)
                
                # Forward pass
                if isinstance(self.model.model, nn.Module) and hasattr(self.model.model, 'forward'):
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        class_output = outputs[0]
                    else:
                        class_output = outputs
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        class_output = outputs[0]
                    else:
                        class_output = outputs
                
                # Get predictions
                probabilities = torch.softmax(class_output, dim=1)
                _, predictions = torch.max(class_output, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(class_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "confusion_matrix": cm.tolist(),
            "domain": domain_name
        }
        
        return metrics
    
    def evaluate_domain_classification(
        self,
        data_loader: torch.utils.data.DataLoader,
        domain_name: str = "unknown"
    ) -> Dict[str, float]:
        """Evaluate domain classification performance (for DANN).
        
        Args:
            data_loader: Data loader
            domain_name: Name of the domain
            
        Returns:
            Dictionary of domain classification metrics
        """
        if self.model.method != "dann":
            return {"domain_accuracy": 0.0, "domain_f1": 0.0}
        
        all_domain_predictions = []
        all_domain_labels = []
        
        with torch.no_grad():
            for images, class_labels, domain_labels in data_loader:
                images = images.to(self.device)
                domain_labels = domain_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, alpha=0.0)  # No gradient reversal
                if isinstance(outputs, tuple):
                    _, domain_output = outputs
                else:
                    continue
                
                # Get domain predictions
                _, domain_predictions = torch.max(domain_output, 1)
                
                all_domain_predictions.extend(domain_predictions.cpu().numpy())
                all_domain_labels.extend(domain_labels.cpu().numpy())
        
        # Calculate domain classification metrics
        domain_accuracy = accuracy_score(all_domain_labels, all_domain_predictions)
        _, _, domain_f1, _ = precision_recall_fscore_support(
            all_domain_labels, all_domain_predictions, average='weighted'
        )
        
        return {
            "domain_accuracy": domain_accuracy,
            "domain_f1": domain_f1,
            "domain": domain_name
        }
    
    def evaluate_domain_generalization(
        self,
        source_loaders: Dict[str, torch.utils.data.DataLoader],
        target_loaders: Dict[str, torch.utils.data.DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate domain generalization performance.
        
        Args:
            source_loaders: Source domain data loaders
            target_loaders: Target domain data loaders
            
        Returns:
            Dictionary of evaluation results
        """
        results = {}
        
        # Evaluate on source domains
        self.logger.info("Evaluating on source domains...")
        for domain_name, loader in source_loaders.items():
            self.logger.info(f"Evaluating source domain: {domain_name}")
            
            # Classification metrics
            class_metrics = self.evaluate_classification(loader, f"source_{domain_name}")
            
            # Domain classification metrics
            domain_metrics = self.evaluate_domain_classification(loader, f"source_{domain_name}")
            
            results[f"source_{domain_name}"] = {**class_metrics, **domain_metrics}
        
        # Evaluate on target domains
        self.logger.info("Evaluating on target domains...")
        for domain_name, loader in target_loaders.items():
            self.logger.info(f"Evaluating target domain: {domain_name}")
            
            # Classification metrics
            class_metrics = self.evaluate_classification(loader, f"target_{domain_name}")
            
            # Domain classification metrics
            domain_metrics = self.evaluate_domain_classification(loader, f"target_{domain_name}")
            
            results[f"target_{domain_name}"] = {**class_metrics, **domain_metrics}
        
        return results
    
    def calculate_domain_gap(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate domain gap metrics.
        
        Args:
            results: Evaluation results
            
        Returns:
            Dictionary of domain gap metrics
        """
        source_results = {k: v for k, v in results.items() if k.startswith("source_")}
        target_results = {k: v for k, v in results.items() if k.startswith("target_")}
        
        if not source_results or not target_results:
            return {}
        
        # Calculate average source and target performance
        source_acc = np.mean([r["accuracy"] for r in source_results.values()])
        target_acc = np.mean([r["accuracy"] for r in target_results.values()])
        
        source_f1 = np.mean([r["f1_score"] for r in source_results.values()])
        target_f1 = np.mean([r["f1_score"] for r in target_results.values()])
        
        # Calculate domain gap
        accuracy_gap = source_acc - target_acc
        f1_gap = source_f1 - target_f1
        
        return {
            "source_accuracy": source_acc,
            "target_accuracy": target_acc,
            "accuracy_gap": accuracy_gap,
            "source_f1": source_f1,
            "target_f1": target_f1,
            "f1_gap": f1_gap,
            "generalization_score": target_acc / source_acc if source_acc > 0 else 0.0
        }
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Create a leaderboard from evaluation results.
        
        Args:
            results: Evaluation results
            save_path: Path to save leaderboard
            
        Returns:
            DataFrame with leaderboard
        """
        leaderboard_data = []
        
        for domain, metrics in results.items():
            leaderboard_data.append({
                "Domain": domain,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "Domain Accuracy": metrics.get("domain_accuracy", 0.0),
                "Domain F1": metrics.get("domain_f1", 0.0)
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values("Accuracy", ascending=False)
        
        if save_path:
            leaderboard.to_csv(save_path, index=False)
            self.logger.info(f"Leaderboard saved to {save_path}")
        
        return leaderboard
    
    def plot_confusion_matrices(
        self,
        results: Dict[str, Dict[str, float]],
        class_names: List[str],
        save_path: Optional[Path] = None
    ) -> None:
        """Plot confusion matrices for all domains.
        
        Args:
            results: Evaluation results
            class_names: List of class names
            save_path: Path to save plots
        """
        num_domains = len(results)
        fig, axes = plt.subplots(2, (num_domains + 1) // 2, figsize=(5 * ((num_domains + 1) // 2), 10))
        
        if num_domains == 1:
            axes = [axes]
        elif num_domains <= 2:
            axes = axes.reshape(1, -1)
        
        for idx, (domain, metrics) in enumerate(results.items()):
            row = idx // ((num_domains + 1) // 2)
            col = idx % ((num_domains + 1) // 2)
            
            cm = np.array(metrics["confusion_matrix"])
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[row, col]
            )
            axes[row, col].set_title(f"{domain}\nAccuracy: {metrics['accuracy']:.3f}")
            axes[row, col].set_xlabel("Predicted")
            axes[row, col].set_ylabel("Actual")
        
        # Hide empty subplots
        for idx in range(num_domains, len(axes.flat)):
            axes.flat[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_domain_generalization_results(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None
    ) -> None:
        """Plot domain generalization results.
        
        Args:
            results: Evaluation results
            save_path: Path to save plot
        """
        domains = list(results.keys())
        accuracies = [results[domain]["accuracy"] for domain in domains]
        
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy by domain
        plt.subplot(1, 2, 1)
        bars = plt.bar(domains, accuracies)
        plt.title("Accuracy by Domain")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        
        # Color bars based on source/target
        for i, (bar, domain) in enumerate(zip(bars, domains)):
            if domain.startswith("source_"):
                bar.set_color('blue')
            else:
                bar.set_color('red')
        
        # Plot domain gap
        plt.subplot(1, 2, 2)
        source_domains = [d for d in domains if d.startswith("source_")]
        target_domains = [d for d in domains if d.startswith("target_")]
        
        if source_domains and target_domains:
            source_acc = np.mean([results[d]["accuracy"] for d in source_domains])
            target_acc = np.mean([results[d]["accuracy"] for d in target_domains])
            
            plt.bar(["Source", "Target"], [source_acc, target_acc], color=['blue', 'red'])
            plt.title(f"Domain Gap: {source_acc - target_acc:.3f}")
            plt.ylabel("Average Accuracy")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Domain generalization plot saved to {save_path}")
        
        plt.show()
    
    def save_results(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Path
    ) -> None:
        """Save evaluation results to JSON.
        
        Args:
            results: Evaluation results
            save_path: Path to save results
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {save_path}")


class EfficiencyEvaluator:
    """Evaluator for model efficiency metrics."""
    
    def __init__(self, device: torch.device):
        """Initialize efficiency evaluator.
        
        Args:
            device: Device to evaluate on
        """
        self.device = device
    
    def evaluate_model_efficiency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Evaluate model efficiency.
        
        Args:
            model: Model to evaluate
            input_shape: Input tensor shape
            num_runs: Number of runs for timing
            
        Returns:
            Dictionary of efficiency metrics
        """
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate FLOPs (simplified)
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Timing
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "average_inference_time": avg_time,
            "fps": fps,
            "input_shape": input_shape
        }
