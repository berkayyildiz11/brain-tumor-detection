import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BaselineCNN, self).__init__()
        # Input shape: (Batch_Size, 3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Spatial dimensions after 3 pooling layers: 224 -> 112 -> 56 -> 28
        # Output features: 128 channels * 28 width * 28 height
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply convolution -> activation -> pooling for each block
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten all dimensions except the batch size
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Quick test to ensure everything is wired up correctly
    model = BaselineCNN(num_classes=4)
    print(model)
    
    # Create a dummy batch identical to what the dataloaders yield
    dummy_input = torch.randn(32, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape} -> Output shape: {output.shape}")