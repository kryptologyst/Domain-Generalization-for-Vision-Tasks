#!/usr/bin/env python3
"""Main training script for domain generalization."""

import argparse
import logging
from pathlib import Path
import sys
import torch
import wandb
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, setup_logging
from models import DomainGeneralizationModel
from data import DomainGeneralizationDataModule
from train import DomainGeneralizationTrainer
from eval import DomainGeneralizationEvaluator, EfficiencyEvaluator


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Domain Generalization Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--test-only", action="store_true", help="Test only mode")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup device
    if config.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    # Setup logging
    logger = setup_logging(config.logging.log_dir, config.logging.level)
    logger.info(f"Starting domain generalization training with config: {config}")
    
    # Initialize wandb if enabled
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            config=OmegaConf.to_container(config, resolve=True),
            name=f"dg_{config.model.method}_{config.model.backbone}"
        )
    
    try:
        # Setup data module
        data_module = DomainGeneralizationDataModule(
            data_dir=config.data.data_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            image_size=config.data.image_size,
            augmentation_strength=config.data.augmentation_strength
        )
        
        # Setup domain generalization data loaders
        data_loaders = data_module.setup_domain_generalization(
            source_domains=config.domains.source,
            target_domains=config.domains.target,
            val_split=config.data.val_split
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(data_loaders['train'])} batches")
        logger.info(f"  Val: {len(data_loaders['val'])} batches")
        logger.info(f"  Target: {len(data_loaders['target'])} batches")
        
        # Initialize model
        model = DomainGeneralizationModel(
            num_classes=config.model.num_classes,
            feature_dim=config.model.feature_dim,
            backbone=config.model.backbone,
            pretrained=config.model.pretrained,
            method=config.model.method,
            alpha=config.model.alpha
        )
        
        logger.info(f"Model initialized: {config.model.method} with {config.model.backbone}")
        
        # Initialize trainer
        trainer = DomainGeneralizationTrainer(
            model=model,
            data_module=data_module,
            config=config.training,
            device=device,
            logger=logger
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.current_epoch = checkpoint["epoch"]
            trainer.best_val_acc = checkpoint["metrics"]["accuracy"]
        
        # Create results directory
        results_dir = Path(config.evaluation.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if not args.test_only:
            # Train the model
            logger.info("Starting training...")
            history = trainer.train(
                train_loader=data_loaders["train"],
                val_loader=data_loaders["val"],
                test_loader=data_loaders["target"],
                save_dir=results_dir
            )
            
            logger.info("Training completed!")
            
            # Save training history
            import json
            with open(results_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)
        
        # Evaluate the model
        logger.info("Starting evaluation...")
        
        # Initialize evaluator
        evaluator = DomainGeneralizationEvaluator(
            model=model,
            device=device,
            logger=logger
        )
        
        # Evaluate on all domains
        source_loaders = {
            domain: data_module.get_single_domain_loaders(domain)["test"]
            for domain in config.domains.source
        }
        
        target_loaders = {
            domain: data_module.get_single_domain_loaders(domain)["test"]
            for domain in config.domains.target
        }
        
        # Run evaluation
        results = evaluator.evaluate_domain_generalization(source_loaders, target_loaders)
        
        # Calculate domain gap
        domain_gap = evaluator.calculate_domain_gap(results)
        logger.info(f"Domain gap metrics: {domain_gap}")
        
        # Create leaderboard
        class_names = [f"Class_{i}" for i in range(config.model.num_classes)]
        leaderboard = evaluator.create_leaderboard(results, results_dir / "leaderboard.csv")
        logger.info("Leaderboard created")
        
        # Create plots
        if config.evaluation.create_plots:
            evaluator.plot_confusion_matrices(
                results, class_names, results_dir / "confusion_matrices.png"
            )
            evaluator.plot_domain_generalization_results(
                results, results_dir / "domain_generalization_results.png"
            )
        
        # Save results
        if config.evaluation.save_results:
            evaluator.save_results(results, results_dir / "evaluation_results.json")
        
        # Evaluate efficiency
        efficiency_evaluator = EfficiencyEvaluator(device)
        efficiency_metrics = efficiency_evaluator.evaluate_model_efficiency(
            model, (3, config.data.image_size, config.data.image_size)
        )
        
        logger.info(f"Efficiency metrics: {efficiency_metrics}")
        
        # Save efficiency metrics
        import json
        with open(results_dir / "efficiency_metrics.json", "w") as f:
            json.dump(efficiency_metrics, f, indent=2)
        
        # Log final results to wandb
        if config.logging.use_wandb:
            wandb.log({
                "final/target_accuracy": domain_gap.get("target_accuracy", 0.0),
                "final/domain_gap": domain_gap.get("accuracy_gap", 0.0),
                "final/generalization_score": domain_gap.get("generalization_score", 0.0),
                "efficiency/fps": efficiency_metrics["fps"],
                "efficiency/parameters": efficiency_metrics["total_parameters"]
            })
        
        logger.info("Evaluation completed!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        if config.logging.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
