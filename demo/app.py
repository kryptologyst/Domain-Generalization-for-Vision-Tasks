"""Streamlit demo for domain generalization."""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import sys
from pathlib import Path
import json
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import get_device, set_seed
from models import DomainGeneralizationModel
from data import DomainGeneralizationDataModule
from eval import DomainGeneralizationEvaluator


# Page configuration
st.set_page_config(
    page_title="Domain Generalization Demo",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .domain-card {
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str, config: dict):
    """Load a trained model."""
    device = get_device()
    model = DomainGeneralizationModel(**config)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model.to(device)
    else:
        st.warning(f"Model checkpoint not found at {model_path}")
        return None


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    data_module = DomainGeneralizationDataModule()
    
    # Load sample images from different domains
    sample_data = {}
    
    try:
        # CIFAR-10 samples
        cifar_loader = data_module.get_single_domain_loaders("cifar10")["test"]
        sample_data["cifar10"] = next(iter(cifar_loader))
        
        # SVHN samples
        svhn_loader = data_module.get_single_domain_loaders("svhn")["test"]
        sample_data["svhn"] = next(iter(svhn_loader))
        
        # MNIST samples
        mnist_loader = data_module.get_single_domain_loaders("mnist")["test"]
        sample_data["mnist"] = next(iter(mnist_loader))
        
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return {}
    
    return sample_data


def preprocess_image(image: Image.Image, target_size: int = 224) -> torch.Tensor:
    """Preprocess uploaded image."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((target_size, target_size))
    
    # Convert to tensor and normalize
    image_array = np.array(image) / 255.0
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).float()
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0)


def predict_image(model, image_tensor, device):
    """Predict class and domain for an image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        if model.method == "dann":
            class_output, domain_output = model(image_tensor)
            
            # Get class prediction
            class_probs = F.softmax(class_output, dim=1)
            class_pred = torch.argmax(class_probs, dim=1)
            
            # Get domain prediction
            domain_probs = F.softmax(domain_output, dim=1)
            domain_pred = torch.argmax(domain_probs, dim=1)
            
            return {
                "class_prediction": class_pred.item(),
                "class_probabilities": class_probs.cpu().numpy()[0],
                "domain_prediction": domain_pred.item(),
                "domain_probabilities": domain_probs.cpu().numpy()[0]
            }
        else:
            class_output = model(image_tensor)
            class_probs = F.softmax(class_output, dim=1)
            class_pred = torch.argmax(class_probs, dim=1)
            
            return {
                "class_prediction": class_pred.item(),
                "class_probabilities": class_probs.cpu().numpy()[0],
                "domain_prediction": None,
                "domain_probabilities": None
            }


def plot_predictions(predictions, class_names):
    """Plot prediction results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Class probabilities
    axes[0].bar(class_names, predictions["class_probabilities"])
    axes[0].set_title("Class Probabilities")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Probability")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Domain probabilities (if available)
    if predictions["domain_probabilities"] is not None:
        domain_names = ["Source", "Target"]
        axes[1].bar(domain_names, predictions["domain_probabilities"])
        axes[1].set_title("Domain Probabilities")
        axes[1].set_xlabel("Domain")
        axes[1].set_ylabel("Probability")
    else:
        axes[1].text(0.5, 0.5, "Domain prediction\nnot available", 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Domain Probabilities")
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🌐 Domain Generalization Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_method = st.sidebar.selectbox(
        "Domain Generalization Method",
        ["dann", "coral", "mixstyle"],
        help="Choose the domain generalization method"
    )
    
    model_backbone = st.sidebar.selectbox(
        "Backbone Architecture",
        ["resnet18", "resnet50"],
        help="Choose the backbone architecture"
    )
    
    # Model configuration
    model_config = {
        "num_classes": 10,
        "feature_dim": 512,
        "backbone": model_backbone,
        "pretrained": True,
        "method": model_method,
        "alpha": 1.0
    }
    
    # Load model
    model_path = f"results/checkpoint.pth"
    model = load_model(model_path, model_config)
    
    if model is None:
        st.error("Please train a model first using the training script.")
        st.code("python scripts/train.py --config configs/config.yaml")
        return
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Image Classification", "📊 Domain Analysis", "📈 Results", "ℹ️ About"])
    
    with tab1:
        st.header("Image Classification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image to classify"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                # Make prediction
                device = get_device()
                predictions = predict_image(model, image_tensor, device)
                
                # Display results
                st.subheader("Prediction Results")
                
                class_names = [f"Class {i}" for i in range(10)]
                predicted_class = predictions["class_prediction"]
                confidence = predictions["class_probabilities"][predicted_class]
                
                st.success(f"Predicted Class: **{class_names[predicted_class]}** (Confidence: {confidence:.3f})")
                
                if predictions["domain_prediction"] is not None:
                    domain_names = ["Source", "Target"]
                    predicted_domain = predictions["domain_prediction"]
                    domain_confidence = predictions["domain_probabilities"][predicted_domain]
                    st.info(f"Predicted Domain: **{domain_names[predicted_domain]}** (Confidence: {domain_confidence:.3f})")
        
        with col2:
            st.subheader("Sample Images")
            
            # Load sample data
            sample_data = load_sample_data()
            
            if sample_data:
                domain_select = st.selectbox("Select Domain", list(sample_data.keys()))
                
                if domain_select in sample_data:
                    images, labels, domain_labels = sample_data[domain_select]
                    
                    # Show first few images
                    for i in range(min(3, len(images))):
                        image = images[i]
                        label = labels[i].item()
                        
                        # Denormalize for display
                        image_display = image.clone()
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        image_display = image_display * std + mean
                        image_display = torch.clamp(image_display, 0, 1)
                        
                        # Convert to PIL
                        image_np = image_display.permute(1, 2, 0).numpy()
                        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                        
                        col_img, col_pred = st.columns([2, 1])
                        
                        with col_img:
                            st.image(image_pil, caption=f"True Label: {label}", width=100)
                        
                        with col_pred:
                            # Make prediction
                            pred = predict_image(model, image.unsqueeze(0), get_device())
                            pred_class = pred["class_prediction"]
                            confidence = pred["class_probabilities"][pred_class]
                            
                            st.write(f"Pred: {pred_class}")
                            st.write(f"Conf: {confidence:.3f}")
                            
                            if pred["domain_prediction"] is not None:
                                domain_names = ["Source", "Target"]
                                pred_domain = domain_names[pred["domain_prediction"]]
                                st.write(f"Domain: {pred_domain}")
    
    with tab2:
        st.header("Domain Analysis")
        
        # Load sample data for analysis
        sample_data = load_sample_data()
        
        if sample_data:
            # Domain comparison
            st.subheader("Domain Comparison")
            
            domain_results = {}
            class_names = [f"Class {i}" for i in range(10)]
            
            for domain_name, (images, labels, domain_labels) in sample_data.items():
                # Take first batch for analysis
                batch_images = images[:32]  # Limit to 32 images
                batch_labels = labels[:32]
                
                # Make predictions
                device = get_device()
                predictions = []
                
                with torch.no_grad():
                    for img in batch_images:
                        img_tensor = img.unsqueeze(0).to(device)
                        pred = predict_image(model, img_tensor, device)
                        predictions.append(pred["class_prediction"])
                
                # Calculate accuracy
                correct = sum(1 for p, l in zip(predictions, batch_labels) if p == l.item())
                accuracy = correct / len(batch_labels)
                
                domain_results[domain_name] = {
                    "accuracy": accuracy,
                    "predictions": predictions,
                    "labels": batch_labels.tolist()
                }
            
            # Plot domain comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            domains = list(domain_results.keys())
            accuracies = [domain_results[d]["accuracy"] for d in domains]
            
            bars = ax.bar(domains, accuracies)
            ax.set_title("Accuracy by Domain")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            
            fig, axes = plt.subplots(1, len(domains), figsize=(5*len(domains), 4))
            if len(domains) == 1:
                axes = [axes]
            
            for i, domain in enumerate(domains):
                predictions = domain_results[domain]["predictions"]
                labels = domain_results[domain]["labels"]
                
                cm = np.zeros((10, 10))
                for pred, label in zip(predictions, labels):
                    cm[label][pred] += 1
                
                sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=axes[i])
                axes[i].set_title(f"{domain}\nAccuracy: {domain_results[domain]['accuracy']:.3f}")
                axes[i].set_xlabel("Predicted")
                axes[i].set_ylabel("Actual")
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab3:
        st.header("Results")
        
        # Check if results exist
        results_dir = Path("results")
        
        if results_dir.exists():
            # Load evaluation results
            results_file = results_dir / "evaluation_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                st.subheader("Evaluation Results")
                
                # Create results table
                results_data = []
                for domain, metrics in results.items():
                    results_data.append({
                        "Domain": domain,
                        "Accuracy": f"{metrics['accuracy']:.3f}",
                        "Precision": f"{metrics['precision']:.3f}",
                        "Recall": f"{metrics['recall']:.3f}",
                        "F1-Score": f"{metrics['f1_score']:.3f}"
                    })
                
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Load leaderboard
                leaderboard_file = results_dir / "leaderboard.csv"
                if leaderboard_file.exists():
                    st.subheader("Leaderboard")
                    leaderboard_df = pd.read_csv(leaderboard_file)
                    st.dataframe(leaderboard_df, use_container_width=True)
                
                # Load efficiency metrics
                efficiency_file = results_dir / "efficiency_metrics.json"
                if efficiency_file.exists():
                    st.subheader("Efficiency Metrics")
                    with open(efficiency_file, 'r') as f:
                        efficiency = json.load(f)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Parameters", f"{efficiency['total_parameters']:,}")
                    
                    with col2:
                        st.metric("FPS", f"{efficiency['fps']:.1f}")
                    
                    with col3:
                        st.metric("Inference Time", f"{efficiency['average_inference_time']*1000:.1f} ms")
            else:
                st.info("No evaluation results found. Please run the training script first.")
        else:
            st.info("No results directory found. Please run the training script first.")
    
    with tab4:
        st.header("About Domain Generalization")
        
        st.markdown("""
        ## What is Domain Generalization?
        
        Domain generalization is a machine learning technique that aims to improve model performance 
        across multiple domains by learning domain-invariant features. Unlike domain adaptation, 
        domain generalization doesn't require access to target domain data during training.
        
        ## Methods Implemented
        
        ### 1. Domain-Adversarial Neural Network (DANN)
        - Uses adversarial training to learn domain-invariant features
        - Includes a domain classifier with gradient reversal
        - Balances classification and domain confusion losses
        
        ### 2. CORAL (CORrelation ALignment)
        - Aligns second-order statistics between domains
        - Minimizes the difference in covariance matrices
        - Simpler approach without adversarial training
        
        ### 3. MixStyle
        - Mixes feature statistics across domains
        - Applies style mixing during training
        - Encourages domain-invariant representations
        
        ## How to Use This Demo
        
        1. **Upload an Image**: Use the file uploader to test on your own images
        2. **View Sample Images**: See how the model performs on different domains
        3. **Analyze Results**: Check the domain analysis and evaluation results
        4. **Compare Methods**: Switch between different domain generalization methods
        
        ## Training Your Own Model
        
        To train a model with your own data:
        
        ```bash
        python scripts/train.py --config configs/config.yaml
        ```
        
        Modify the configuration file to adjust:
        - Model architecture and method
        - Training parameters
        - Domain selection
        - Data augmentation strength
        """
        )
        
        st.subheader("Technical Details")
        
        st.markdown("""
        - **Framework**: PyTorch 2.0+
        - **Backbones**: ResNet-18, ResNet-50
        - **Datasets**: CIFAR-10, SVHN, MNIST, Fashion-MNIST
        - **Augmentation**: Albumentations with configurable strength
        - **Evaluation**: Comprehensive metrics including domain gap analysis
        """)


if __name__ == "__main__":
    main()
