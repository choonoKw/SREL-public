# -*- coding: utf-8 -*-
"""
Created on Tue April 2 2024

@author: jbk5816
"""

from model.estimate_rho import Estimate_rho
from utils.functions import eta_sred


import torch
import torch.nn as nn

class SREL_intra_phase1_vary_rho(nn.Module):
    def __init__(self, constants):
        super(SREL_intra_phase1_vary_rho, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        # Dynamically create the modules for estimating eta and rho
        self.est_rho_modules = nn.ModuleList([
                    Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
                    for _ in range(self.N_step)
                ])
        
        
    def forward(self, phi_batch, w_batch, y, G_batch, H_batch):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
        rho_stack_batch = torch.zeros(batch_size, N_step).to(self.device)
        
        # phi_batch = phi0_batch
        
        # Repeat the update process N_step times
        for update_step in range(N_step):
            s_batch = modulus*torch.exp(1j *phi_batch)                    
            y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
            
            x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=1)
            
            eta_batch = eta_sred(G_batch, H_batch, s_batch)


            rho_batch = self.est_rho_modules[update_step](x_batch)

            rho_stack_batch[:,update_step] = rho_batch.squeeze()
    
            # Update phi
            phi_batch = phi_batch - rho_batch*eta_batch  
    
            # save on list
            s_stack_batch[:,update_step,:] = s_batch
    
        s_stack_batch[:,N_step,:] = modulus*torch.exp(1j *phi_batch)  # Saving the final s after all updates
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'rho_stack_batch': rho_stack_batch
        }
        return model_outputs
    
    
    
class SREL_intra_phase1_rep_rho(nn.Module):
    def __init__(self, constants):
        super(SREL_intra_phase1_rep_rho, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        # Dynamically create the modules for estimating eta and rho
        self.est_rho_modules = Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
        
        
    def forward(self, phi_batch, w_batch, y, G_batch, H_batch):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
        rho_stack_batch = torch.zeros(batch_size, N_step).to(self.device)
        
        # phi_batch = phi0_batch
        
        # Repeat the update process N_step times
        for update_step in range(N_step):
            s_batch = modulus*torch.exp(1j *phi_batch)                    
            y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
            
            x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=1)
            
            eta_batch = eta_sred(G_batch, H_batch, s_batch)


            rho_batch = self.est_rho_modules(x_batch)

            rho_stack_batch[:,update_step] = rho_batch.squeeze()
    
            # Update phi
            phi_batch = phi_batch - rho_batch*eta_batch  
    
            # save on list
            s_stack_batch[:,update_step,:] = s_batch
    
        s_stack_batch[:,N_step,:] = modulus*torch.exp(1j *phi_batch)  # Saving the final s after all updates
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'rho_stack_batch': rho_stack_batch
        }
        return model_outputs