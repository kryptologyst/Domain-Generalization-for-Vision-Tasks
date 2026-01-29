"""Test suite for domain generalization project."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, EarlyStopping, count_parameters
from models import DANN, CORAL, MixStyle, StyleAugment, DomainGeneralizationModel
from data import DomainDataset, DomainGeneralizationDataModule, SyntheticDomainDataset
from train import DomainGeneralizationTrainer
from eval import DomainGeneralizationEvaluator, EfficiencyEvaluator


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        rand1 = torch.rand(1)
        
        set_seed(42)
        rand2 = torch.rand(1)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Linear(10, 5)
        param_count = count_parameters(model)
        assert param_count == 55  # 10*5 + 5 bias
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        model = nn.Linear(10, 1)
        early_stop = EarlyStopping(patience=2, min_delta=0.1)
        
        # Should not stop initially
        assert not early_stop(0.5, model)
        assert not early_stop(0.4, model)
        
        # Should stop after patience exceeded
        assert not early_stop(0.6, model)
        assert not early_stop(0.7, model)
        assert early_stop(0.8, model)


class TestModels:
    """Test model implementations."""
    
    def test_dann_forward(self):
        """Test DANN forward pass."""
        model = DANN(num_classes=10, backbone="resnet18", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        
        class_output, domain_output = model(x)
        
        assert class_output.shape == (2, 10)
        assert domain_output.shape == (2, 2)
    
    def test_coral_forward(self):
        """Test CORAL forward pass."""
        model = CORAL(num_classes=10, backbone="resnet18", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_coral_loss(self):
        """Test CORAL loss computation."""
        model = CORAL(num_classes=10, backbone="resnet18", pretrained=False)
        
        source_features = torch.randn(32, 512)
        target_features = torch.randn(32, 512)
        
        loss = model.coral_loss(source_features, target_features)
        assert loss.item() >= 0
        assert isinstance(loss, torch.Tensor)
    
    def test_mixstyle_forward(self):
        """Test MixStyle forward pass."""
        mixstyle = MixStyle()
        x = torch.randn(4, 3, 224, 224)
        
        output = mixstyle(x)
        assert output.shape == x.shape
    
    def test_style_augment_forward(self):
        """Test StyleAugment forward pass."""
        style_aug = StyleAugment(num_domains=3)
        x = torch.randn(2, 3, 224, 224)
        
        output = style_aug(x)
        assert output.shape == x.shape
    
    def test_domain_generalization_model(self):
        """Test unified domain generalization model."""
        # Test DANN method
        model = DomainGeneralizationModel(method="dann", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        
        outputs = model(x)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        
        # Test CORAL method
        model = DomainGeneralizationModel(method="coral", pretrained=False)
        outputs = model(x)
        assert isinstance(outputs, torch.Tensor)
        
        # Test MixStyle method
        model = DomainGeneralizationModel(method="mixstyle", pretrained=False)
        outputs = model(x)
        assert isinstance(outputs, torch.Tensor)
    
    def test_model_loss_computation(self):
        """Test loss computation for different methods."""
        # Test DANN loss
        model = DomainGeneralizationModel(method="dann", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        targets = torch.randint(0, 10, (2,))
        domain_targets = torch.randint(0, 2, (2,))
        
        outputs = model(x)
        losses = model.get_loss(outputs, targets, domain_labels=domain_targets)
        
        assert "classification" in losses
        assert "domain" in losses
        assert "total" in losses
        
        # Test CORAL loss
        model = DomainGeneralizationModel(method="coral", pretrained=False)
        outputs = model(x)
        losses = model.get_loss(outputs, targets)
        
        assert "classification" in losses
        assert "total" in losses


class TestData:
    """Test data loading and preprocessing."""
    
    def test_domain_dataset(self):
        """Test DomainDataset functionality."""
        # Create dummy datasets
        dataset1 = torch.utils.data.TensorDataset(
            torch.randn(10, 3, 32, 32),
            torch.randint(0, 10, (10,))
        )
        dataset2 = torch.utils.data.TensorDataset(
            torch.randn(8, 3, 32, 32),
            torch.randint(0, 10, (8,))
        )
        
        domain_dataset = DomainDataset([dataset1, dataset2], [0, 1])
        
        assert len(domain_dataset) == 18
        
        # Test indexing
        image, class_label, domain_label = domain_dataset[0]
        assert image.shape == (3, 32, 32)
        assert isinstance(class_label, int)
        assert domain_label == 0
        
        image, class_label, domain_label = domain_dataset[10]
        assert domain_label == 1
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        dataset = SyntheticDomainDataset(
            num_samples=100,
            num_classes=5,
            image_size=64,
            domain_style="cartoon"
        )
        
        assert len(dataset) == 100
        
        image, label = dataset[0]
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert 0 <= label < 5
    
    def test_data_module(self):
        """Test data module functionality."""
        data_module = DomainGeneralizationDataModule(
            batch_size=16,
            image_size=64
        )
        
        # Test single domain loaders
        loaders = data_module.get_single_domain_loaders("cifar10")
        
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders
        
        # Test batch structure
        batch = next(iter(loaders["train"]))
        images, labels = batch
        assert images.shape[0] <= 16  # batch size
        assert images.shape[1:] == (3, 64, 64)
        assert len(labels) == images.shape[0]


class TestTraining:
    """Test training framework."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = DomainGeneralizationModel(method="dann", pretrained=False)
        data_module = DomainGeneralizationDataModule(batch_size=16)
        
        config = {
            "max_epochs": 5,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": None,
            "patience": 3,
            "min_delta": 0.001
        }
        
        device = get_device()
        logger = logging.getLogger(__name__)
        
        trainer = DomainGeneralizationTrainer(
            model, data_module, config, device, logger
        )
        
        assert trainer.model == model
        assert trainer.device == device
        assert trainer.current_epoch == 0
    
    def test_trainer_optimizer_setup(self):
        """Test optimizer setup."""
        model = DomainGeneralizationModel(method="dann", pretrained=False)
        data_module = DomainGeneralizationDataModule()
        
        config = {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-4
        }
        
        device = get_device()
        logger = logging.getLogger(__name__)
        
        trainer = DomainGeneralizationTrainer(
            model, data_module, config, device, logger
        )
        
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.001
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 1e-4


class TestEvaluation:
    """Test evaluation framework."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        model = DomainGeneralizationModel(method="dann", pretrained=False)
        device = get_device()
        logger = logging.getLogger(__name__)
        
        evaluator = DomainGeneralizationEvaluator(model, device, logger)
        
        assert evaluator.model == model
        assert evaluator.device == device
    
    def test_efficiency_evaluator(self):
        """Test efficiency evaluator."""
        model = DomainGeneralizationModel(method="dann", pretrained=False)
        device = get_device()
        
        evaluator = EfficiencyEvaluator(device)
        metrics = evaluator.evaluate_model_efficiency(
            model, (3, 224, 224), num_runs=10
        )
        
        assert "total_parameters" in metrics
        assert "fps" in metrics
        assert "average_inference_time" in metrics
        assert metrics["total_parameters"] > 0
        assert metrics["fps"] > 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training with synthetic data."""
        set_seed(42)
        
        # Create synthetic datasets
        source_dataset = SyntheticDomainDataset(
            num_samples=100,
            num_classes=5,
            domain_style="cartoon"
        )
        target_dataset = SyntheticDomainDataset(
            num_samples=50,
            num_classes=5,
            domain_style="sketch"
        )
        
        # Create domain dataset
        domain_dataset = DomainDataset(
            [source_dataset, target_dataset],
            [0, 1]
        )
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            domain_dataset, batch_size=16, shuffle=True
        )
        
        # Create model
        model = DomainGeneralizationModel(
            method="dann",
            num_classes=5,
            pretrained=False
        )
        
        # Create trainer
        config = {
            "max_epochs": 2,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": None,
            "patience": 10,
            "min_delta": 0.001
        }
        
        device = get_device()
        logger = logging.getLogger(__name__)
        
        trainer = DomainGeneralizationTrainer(
            model, None, config, device, logger
        )
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(data_loader)
        
        assert "loss" in train_metrics
        assert "accuracy" in train_metrics
        assert train_metrics["loss"] > 0
        assert 0 <= train_metrics["accuracy"] <= 100


if __name__ == "__main__":
    pytest.main([__file__])
