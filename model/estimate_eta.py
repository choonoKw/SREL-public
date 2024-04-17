# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:09:07 2024

@author: jbk5816

Estimate descent direction \eta from input x. 
x consists of [real(s), imag(s), real(w_m), imag(w_m), v_m],
where v_m denotes the mapping vector for m-th target
"""

# import torch
import torch.nn as nn


class Estimate_eta(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Estimate_eta, self).__init__()
        # Define layers for estimating descent direction eta
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # Output dimension matches phi 128
        )

    def forward(self, x):
        return self.layers(x)
    
class Estimate_etaL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Estimate_etaL, self).__init__()
        # Define layers for estimating descent direction eta
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # Output dimension matches phi 128
        )

    def forward(self, x):
        return self.layers(x)