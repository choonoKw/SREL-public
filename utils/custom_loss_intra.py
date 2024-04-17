# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:10:00 2024

@author: jbk5816
"""

import torch
import numpy as np

# from utils.custom_loss import reciprocal_sinr, regularizer_eta

from utils.custom_loss_batch import reciprocal_sinr
from utils.functions import eta_sred

# def custom_loss_intra_phase1(constants, G_batch, H_batch, hyperparameters, model_outputs):
#     N_step = constants['N_step']
#     device = G_batch.device
#     s_stack_batch = model_outputs['s_stack_batch'].to(device)
#     rho_stack_batch = model_outputs['rho_stack_batch'].to(device)
#     batch_size = s_stack_batch.size(0)
    
#     f_sinr_sum = 0.0
    
#     for update_step in range(N_step-1):
#         s_batch =  s_stack_batch[:,update_step+1,:]
            
#         # f_sinr = 
            
#         f_sinr_sum += torch.sum(reciprocal_sinr(G_batch, H_batch, s_batch)).item()
        
#         # f_rho_sum += hyperparameters['lambda_var_rho']*var_rho_avg
        
#     s_batch =  s_stack_batch[:,-1,:]
    
#     f_sinr_opt_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
#     f_sinr_opt = torch.sum(f_sinr_opt_batch)
    
#     sinr_opt_avg = torch.sum(1/f_sinr_opt_batch)/batch_size
    
#     var_rho_avg = torch.sum(torch.var(rho_stack_batch, dim=0, unbiased=False))
    
#     loss = (
#         f_sinr_opt
#         + hyperparameters['lambda_sinr']*f_sinr_sum/(N_step-1)
#         + hyperparameters['lambda_var_rho']*var_rho_avg
#         )
    
#     loss_avg = loss / batch_size 
    
#     return loss_avg, sinr_opt_avg

def custom_loss_intra_phase1_mono(constants, G_batch, H_batch, hyperparameters, model_outputs):
    N_step = constants['N_step']
    device = G_batch.device
    s_stack_batch = model_outputs['s_stack_batch'].to(device)
    rho_stack_batch = model_outputs['rho_stack_batch'].to(device)
    batch_size = s_stack_batch.size(0)
    
    f_sinr_sum = 0.0
    
    s_batch =  s_stack_batch[:,0,:]
    f_sinr_t1_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
    
    N_vl_sum = 0 # number of violation of monotonicity
    for update_step in range(N_step):
        s_batch =  s_stack_batch[:,update_step+1,:]
            
        f_sinr_t2_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
        # f_sinr_t2 = torch.sum(reciprocal_sinr(G_batch, H_batch, s_batch)).item()
        f_sinr_sum += torch.sum(
            torch.exp(
                hyperparameters['lambda_mono']*(f_sinr_t2_batch-f_sinr_t1_batch)
            )
        )/batch_size
        
        N_vl_sum += torch.sum(
            (f_sinr_t2_batch - f_sinr_t1_batch > 0).int()
            )
        
    s_batch =  s_stack_batch[:,-1,:]
    
    f_sinr_opt_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
    f_sinr_opt = torch.sum(f_sinr_opt_batch)
    
    sinr_opt_avg = torch.sum(1/f_sinr_opt_batch)/batch_size
    
    var_rho_avg = torch.sum(torch.var(rho_stack_batch, dim=0, unbiased=False))
    
    loss = (
        f_sinr_opt
        + hyperparameters['lambda_sinr']*f_sinr_sum/N_step
        + hyperparameters['lambda_var_rho']*var_rho_avg
        )
    
    loss_avg = loss / batch_size 
    
    N_vl_avg = N_vl_sum / batch_size
    
    return loss_avg, sinr_opt_avg, N_vl_avg

# def custom_loss_intra_phase2(constants, G_batch, H_batch, hyperparameters, model_outputs):
#     N_step = constants['N_step']
#     # device = G_batch.device
#     s_stack_batch = model_outputs['s_stack_batch']
#     eta_stack_batch = model_outputs['eta_stack_batch']
#     batch_size = s_stack_batch.size(0)
    
#     f_sinr_sum = 0.0
#     f_eta_sum = 0.0
    
#     for update_step in range(N_step-1):
#         s_batch =  s_stack_batch[:,update_step+1,:]
            
#         f_sinr_sum += torch.sum(reciprocal_sinr(G_batch, H_batch, s_batch)).item()
        
#         eta_batch = eta_stack_batch[:,update_step,:]
#         eta_tilde_batch = eta_sred(G_batch, H_batch, s_batch)
        
#         f_eta_sum += torch.norm(eta_tilde_batch - eta_batch) ** 2
        
#         # f_rho_sum += hyperparameters['lambda_var_rho']*var_rho_avg
        
#     s_batch =  s_stack_batch[:,-1,:]
    
#     f_sinr_opt_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
#     f_sinr_opt = torch.sum(f_sinr_opt_batch)
    
#     sinr_opt_avg = torch.sum(1/f_sinr_opt_batch)/batch_size
    
#     loss = (
#         f_sinr_opt
#         + hyperparameters['lambda_sinr']*f_sinr_sum/(N_step-1)
#         + hyperparameters['lambda_eta']*f_eta_sum/N_step
#         )
    
#     loss_avg = loss / batch_size 
    
#     return loss_avg, sinr_opt_avg

def custom_loss_intra_phase2_mono(constants, G_batch, H_batch, hyperparameters, model_outputs):
    N_step = constants['N_step']
    # device = G_batch.device
    s_stack_batch = model_outputs['s_stack_batch']
    eta_stack_batch = model_outputs['eta_stack_batch']
    batch_size = s_stack_batch.size(0)
    
    
    f_sinr_sum = 0.0
    f_eta_sum = 0.0
    
    s_batch =  s_stack_batch[:,0,:]
    f_sinr_t1_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
    
    N_vl_sum = 0 # number of violation of monotonicity
    for update_step in range(N_step):
        s_batch =  s_stack_batch[:,update_step+1,:]
            
        f_sinr_t2_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
        # f_sinr_t2 = torch.sum(reciprocal_sinr(G_batch, H_batch, s_batch)).item()
        f_sinr_sum += torch.sum(
            torch.exp(
                hyperparameters['lambda_mono']*(f_sinr_t2_batch-f_sinr_t1_batch)
            )
        )/batch_size
        
        N_vl_sum += torch.sum(
            (f_sinr_t2_batch - f_sinr_t1_batch > 0).int()
            )
        
        eta_batch = eta_stack_batch[:,update_step,:]
        eta_tilde_batch = eta_sred(G_batch, H_batch, s_batch)
        
        f_eta_sum += torch.norm(eta_tilde_batch - eta_batch) ** 2
        
        # f_rho_sum += hyperparameters['lambda_var_rho']*var_rho_avg
        
    s_batch =  s_stack_batch[:,-1,:]
    
    f_sinr_opt_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
    f_sinr_opt = torch.sum(f_sinr_opt_batch)
    
    sinr_opt_avg = torch.sum(1/f_sinr_opt_batch)/batch_size
    
    loss = (
        f_sinr_opt
        + hyperparameters['lambda_sinr']*f_sinr_sum/(N_step-1)
        + hyperparameters['lambda_eta']*f_eta_sum/N_step
        )
    
    loss_avg = loss / batch_size 
    
    return loss_avg, sinr_opt_avg


# def custom_loss_function(constants, G_batch, H_batch, hyperparameters, model_outputs):
#     N_step = constants['N_step']
#     s_stack_batch = model_outputs['s_stack_batch']
#     s_stack_batch = s_stack_batch.to(G_batch.device)
#     eta_stack_batch = model_outputs['eta_stack_batch']
#     eta_stack_batch = eta_stack_batch.to (G_batch.device)
#     batch_size = s_stack_batch.size(0)
    
#     loss_sum = 0.0
    
#     for idx_batch in range(batch_size):
#         G = G_batch[idx_batch]
#         H = H_batch[idx_batch]
        
#         s_stack = s_stack_batch[idx_batch]
#         eta_stack = eta_stack_batch[idx_batch]
        
        
#         s = s_stack[0]
#         eta = eta_stack[0]
    
#         f_eta = regularizer_eta(G, H, s, eta)
#         f_sinr = 0.0
    
#         for n in range(N_step-1):
#             s = s_stack[n+1]
#             eta = eta_stack[n+1]
            
#             f_eta += regularizer_eta(G, H, s, eta)
#             f_sinr += reciprocal_sinr(G, H, s)
        
#         s = s_stack[N_step]
#         f_sinr_opt = reciprocal_sinr(G, H, s)
    
#         loss = f_sinr_opt + \
#             hyperparameters['lambda_sinr']*f_sinr/(N_step-1) + hyperparameters['lambda_eta']*f_eta/N_step
            
#         loss_sum += loss
    
#     return loss_sum/ batch_size