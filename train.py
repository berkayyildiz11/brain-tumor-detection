import torch
import torch.nn as nn
import torch.optim as optim

# Import your model and your friend's data pipeline
from baseline_cnn import BaselineCNN
from src.data.dataloaders import create_dataloaders

def main():
    # 1. Setup device (Apple Silicon MPS, Nvidia CUDA, or CPU fallback)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Load the data using the provided dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders()
    
    # 3. Initialize Model, Loss Function, and Optimizer
    model = BaselineCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    best_val_acc = 0.0
    print("\nStarting training...")
    
    # 4. The Training and Validation Loop
    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()       # Clear old gradients
            outputs = model(images)     # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward()             # Backward pass
            optimizer.step()            # Update weights
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad(): # Don't track gradients during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {100 * train_correct / train_total:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Accuracy: {val_acc:.2f}%")
              
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_baseline_cnn.pth")
            print(">>> Best model saved!")

if __name__ == "__main__":
    main()