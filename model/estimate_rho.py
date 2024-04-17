# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:09:07 2024

@author: jbk5816

Layers to estimate rho (step size for descent update)
"""

# import torch
import torch.nn as nn
import torch.nn.functional as F

class Estimate_rho(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Estimate_rho, self).__init__()
        # Define layers for estimating step size rho
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),  # Start with a more significant reduction
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),  # Optional: add dropout for regularization
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),  # Output dimension matches rho 1
        )

    def forward(self, x):
        y = self.layers(x)
        # Apply Softplus to ensure positive output
        return F.softplus(y)
    
class Estimate_rho_DO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Estimate_rho_DO, self).__init__()
        # Define layers for estimating step size rho
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),  # Start with a more significant reduction
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Optional: add dropout for regularization
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),  # Output dimension matches rho 1
        )

    def forward(self, x):
        y = self.layers(x)
        # Apply Softplus to ensure positive output
        return F.softplus(y)
    
# layers with batch normalization
class Estimate_rho_BN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Estimate_rho_BN, self).__init__()
        # Define layers for estimating step size rho
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),  # Start with a more significant reduction
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_dim),  # Output dimension matches rho 1
        )

    def forward(self, x):
        y = self.layers(x)
        # Apply Softplus to ensure positive output
        return F.softplus(y)