# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AorticDissectionClassifier(nn.Module):
    def __init__(self):
        super(AorticDissectionClassifier, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16 * 8, 128)  # Assuming input (1, 64, 64, 32)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, D/2, H/2, W/2]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, D/4, H/4, W/4]
        x = self.pool(F.relu(self.conv3(x)))   # [B, 128, D/8, H/8, W/8]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x
