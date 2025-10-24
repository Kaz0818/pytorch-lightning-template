"""
シンプルなCNNモデル（Step 1-3で使用）
"""
import torch.nn as nn


class SimpleCNN(nn.Module):
    """シンプルな3層CNN（96x96のSTL-10用）"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 96x96x3 -> 48x48x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 48x48x32 -> 24x24x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 24x24x64 -> 12x12x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
