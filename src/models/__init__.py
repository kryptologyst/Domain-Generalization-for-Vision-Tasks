"""Domain generalization models implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any
import math


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training."""
    
    @staticmethod
    def forward(ctx: torch.autograd.Function, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward pass."""
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx: torch.autograd.Function, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass with gradient reversal."""
        return grad_output.neg() * ctx.alpha, None


class DANN(nn.Module):
    """Domain-Adversarial Neural Network for domain generalization."""
    
    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = True,
        alpha: float = 1.0
    ):
        """Initialize DANN model.
        
        Args:
            num_classes: Number of classification classes
            feature_dim: Feature dimension
            backbone: Backbone architecture
            pretrained: Whether to use pretrained weights
            alpha: Gradient reversal strength
        """
        super(DANN, self).__init__()
        self.alpha = alpha
        
        # Feature extractor
        if backbone == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            self.feature_dim = 512
        elif backbone == "resnet50":
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 2)  # Binary domain classification
        )
    
    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor
            alpha: Gradient reversal strength
            
        Returns:
            Tuple of (class_output, domain_output)
        """
        if alpha is None:
            alpha = self.alpha
            
        # Extract features
        features = self.feature_extractor(x)
        
        # Classification
        class_output = self.classifier(features)
        
        # Domain classification with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        
        return class_output, domain_output


class CORAL(nn.Module):
    """CORAL (CORrelation ALignment) for domain adaptation."""
    
    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = True
    ):
        """Initialize CORAL model.
        
        Args:
            num_classes: Number of classification classes
            feature_dim: Feature dimension
            backbone: Backbone architecture
            pretrained: Whether to use pretrained weights
        """
        super(CORAL, self).__init__()
        
        # Feature extractor
        if backbone == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            self.feature_dim = 512
        elif backbone == "resnet50":
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Classification output
        """
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def coral_loss(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute CORAL loss between source and target features.
        
        Args:
            source_features: Source domain features
            target_features: Target domain features
            
        Returns:
            CORAL loss
        """
        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        # Frobenius norm of difference
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        loss = loss / (4 * source_features.size(1) ** 2)
        
        return loss
    
    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix of features.
        
        Args:
            features: Input features
            
        Returns:
            Covariance matrix
        """
        # Center the features
        centered = features - features.mean(dim=0, keepdim=True)
        
        # Compute covariance
        cov = torch.mm(centered.t(), centered) / (features.size(0) - 1)
        
        return cov


class MixStyle(nn.Module):
    """MixStyle module for domain generalization."""
    
    def __init__(self, alpha: float = 0.1, eps: float = 1e-6):
        """Initialize MixStyle.
        
        Args:
            alpha: Mixing strength
            eps: Small value for numerical stability
        """
        super(MixStyle, self).__init__()
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MixStyle to input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Styled tensor
        """
        batch_size = x.size(0)
        
        # Compute mean and std
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        
        # Random permutation for mixing
        perm = torch.randperm(batch_size)
        
        # Mix statistics
        mean_mix = self.alpha * mean + (1 - self.alpha) * mean[perm]
        std_mix = self.alpha * std + (1 - self.alpha) * std[perm]
        
        # Denormalize
        x_mix = x_norm * std_mix + mean_mix
        
        return x_mix


class StyleAugment(nn.Module):
    """Style augmentation for domain generalization."""
    
    def __init__(self, num_domains: int = 4):
        """Initialize StyleAugment.
        
        Args:
            num_domains: Number of style domains
        """
        super(StyleAugment, self).__init__()
        self.num_domains = num_domains
        self.style_networks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1)
            ) for _ in range(num_domains)
        ])
    
    def forward(self, x: torch.Tensor, domain_idx: Optional[int] = None) -> torch.Tensor:
        """Apply style augmentation.
        
        Args:
            x: Input tensor
            domain_idx: Specific domain index (if None, random)
            
        Returns:
            Style augmented tensor
        """
        if domain_idx is None:
            domain_idx = torch.randint(0, self.num_domains, (1,)).item()
        
        # Apply style transformation
        style_transform = self.style_networks[domain_idx](x)
        
        # Mix with original
        alpha = torch.rand(1).item()
        x_aug = alpha * x + (1 - alpha) * style_transform
        
        return x_aug


class DomainGeneralizationModel(nn.Module):
    """Unified domain generalization model with multiple strategies."""
    
    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = True,
        method: str = "dann",
        **kwargs
    ):
        """Initialize domain generalization model.
        
        Args:
            num_classes: Number of classes
            feature_dim: Feature dimension
            backbone: Backbone architecture
            pretrained: Whether to use pretrained weights
            method: Domain generalization method ('dann', 'coral', 'mixstyle')
            **kwargs: Additional arguments
        """
        super(DomainGeneralizationModel, self).__init__()
        
        self.method = method
        
        if method == "dann":
            self.model = DANN(
                num_classes=num_classes,
                feature_dim=feature_dim,
                backbone=backbone,
                pretrained=pretrained,
                **kwargs
            )
        elif method == "coral":
            self.model = CORAL(
                num_classes=num_classes,
                feature_dim=feature_dim,
                backbone=backbone,
                pretrained=pretrained
            )
        elif method == "mixstyle":
            self.base_model = models.resnet18(pretrained=pretrained)
            self.base_model.fc = nn.Linear(512, num_classes)
            self.mixstyle = MixStyle()
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Any:
        """Forward pass.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Model output
        """
        if self.method == "mixstyle":
            # Apply MixStyle to intermediate features
            x = self.mixstyle(x)
            return self.base_model(x)
        else:
            return self.model(x, **kwargs)
    
    def get_loss(self, outputs: Any, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute loss based on method.
        
        Args:
            outputs: Model outputs
            targets: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        if self.method == "dann":
            class_output, domain_output = outputs
            source_labels = targets
            domain_labels = kwargs.get("domain_labels", torch.zeros_like(targets))
            
            # Classification loss
            losses["classification"] = F.cross_entropy(class_output, source_labels)
            
            # Domain adversarial loss
            losses["domain"] = F.cross_entropy(domain_output, domain_labels)
            
            # Total loss
            losses["total"] = losses["classification"] + losses["domain"]
            
        elif self.method == "coral":
            class_output = outputs
            losses["classification"] = F.cross_entropy(class_output, targets)
            
            # Add CORAL loss if source and target features provided
            if "source_features" in kwargs and "target_features" in kwargs:
                losses["coral"] = self.model.coral_loss(
                    kwargs["source_features"], 
                    kwargs["target_features"]
                )
                losses["total"] = losses["classification"] + losses["coral"]
            else:
                losses["total"] = losses["classification"]
                
        elif self.method == "mixstyle":
            class_output = outputs
            losses["classification"] = F.cross_entropy(class_output, targets)
            losses["total"] = losses["classification"]
        
        return losses
