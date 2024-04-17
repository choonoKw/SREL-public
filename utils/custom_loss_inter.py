# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:51:20 2024

@author: jbk5816
"""


import torch
from utils.custom_loss_intra import reciprocal_sinr


def custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, s_stack_batch):
    N_step = constants['N_step']
    # s_stack_batch = model_outputs['s_stack_batch']
    # eta_stack_batch = model_outputs['eta_stack_batch']
    batch_size = s_stack_batch.size(0)
    
    loss_sum = 0.0
    
    for idx_batch in range(batch_size):
        G_M = G_M_batch[idx_batch]
        H_M = H_M_batch[idx_batch]
        
        s_stack = s_stack_batch[idx_batch]
        
        f_sinr = 0.0
    
        for n in range(N_step-1):
            s = s_stack[n+1]
            
            for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=2),
                                                    torch.unbind(H_M, dim=2))):
            
                f_sinr += reciprocal_sinr(constants, G, H, s)
        
        # compute the f_sinr with the output s
        s = s_stack[-1]
        f_sinr_opt = 0.0
        for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=2),
                                                torch.unbind(H_M, dim=2))):
            f_sinr_opt += reciprocal_sinr(constants, G, H, s)
    
        loss = f_sinr_opt + hyperparameters['lambda_sinr']*f_sinr/(N_step-1)
            
        loss_sum += loss
    
    return loss_sum/ batch_size