import os
import sys

# Add project root to sys.path so 'src' can be found when running directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from sklearn.metrics import classification_report

# Import your model and data pipeline
from src.cnnModel.baseline_cnn import BaselineCNN
from src.data.dataloaders import create_dataloaders

def main():
    # 1. Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Load the data 
    print("Loading test dataset...")
    # We only need the test_loader and class_names for evaluation
    _, _, test_loader, class_names = create_dataloaders()
    
    # 3. Initialize Model and Load Best Weights
    print("Loading saved model weights...")
    model = BaselineCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("best_baseline_cnn.pth", map_location=device, weights_only=True))
    model.eval() # Set model to evaluation mode
    
    # 4. Evaluation Loop
    print("\nEvaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad(): # No need to track gradients for testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and true labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 5. Calculate and Print Metrics
    print("\n--- Evaluation Results ---")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    main()