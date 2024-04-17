# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:51:22 2024

@author: jbk5816
"""

# from model.estimate_rho_sred import Estimate_rho
from model.estimate_eta import Estimate_eta, Estimate_etaL



# from utils.input_standardize import standardize

import torch
import torch.nn as nn
from torch.nn import ModuleList


class SREL_vary_eta(nn.Module):
    def __init__(self, constants, model_intra_phase1):
        super(SREL_vary_eta, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        self.model_intra_phase1 = model_intra_phase1
        
        # Dynamically create the modules for estimating eta and rho
        self.est_eta_modules = nn.ModuleList([
                    Estimate_eta(2*self.Ls + 2*constants['Lw'] + constants['Ly'], self.Ls)
                    for _ in range(self.N_step)
                ])
        
        
    def forward(self, phi_batch, w_batch, y):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        Ls = self.Ls
        device = self.device
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, Ls, dtype=torch.complex64).to(device)
        eta_stack_batch = torch.zeros(batch_size, N_step, Ls).to(device)
        
        if isinstance(self.model_intra_phase1.est_rho_modules, ModuleList):
            # model_intra_phase1 has various NN modules.
            
            # Repeat the update process N_step times
            for update_step in range(N_step):
                s_batch = modulus*torch.exp(1j *phi_batch)
                y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                
                x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=1)
                    
                eta_batch = self.est_eta_modules[update_step](x_batch)

                rho_batch = self.model_intra_phase1.est_rho_modules[update_step](x_batch)


                eta_stack_batch[:,update_step,:] = eta_batch
    
                # Update phi
                phi_batch = phi_batch - rho_batch*eta_batch  
    
                # save on list
                s_stack_batch[:,update_step,:] = s_batch
                
        else: # model_intra_phase1 has repeated NN module.
            
            # Repeat the update process N_step times
            for update_step in range(N_step):
                s_batch = modulus*torch.exp(1j *phi_batch)
                y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                
                x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=1)
                    
                eta_batch = self.est_eta_modules[update_step](x_batch)

                rho_batch = self.model_intra_phase1.est_rho_modules(x_batch)


                eta_stack_batch[:,update_step,:] = eta_batch
    
                # Update phi
                phi_batch = phi_batch - rho_batch*eta_batch  
    
                # save on list
                s_stack_batch[:,update_step,:] = s_batch
            
        # update along steps finished
        s_stack_batch[:,N_step,:] = modulus*torch.exp(1j *phi_batch)  # Saving the final s after all updates
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'eta_stack_batch': eta_stack_batch
        }
        return model_outputs

    
class SREL_rep_eta(nn.Module):
    def __init__(self, constants, model_intra_phase1):
        super(SREL_rep_eta, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        self.model_intra_phase1 = model_intra_phase1
        
        # Dynamically create the modules for estimating eta and rho
        self.est_eta_modules = Estimate_eta(2*self.Ls + 2*constants['Lw'] + constants['Ly'], self.Ls)
        
        
    def forward(self, phi_batch, w_batch, y):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        Ls = self.Ls
        device = self.device
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, Ls, dtype=torch.complex64).to(device)
        eta_stack_batch = torch.zeros(batch_size, N_step, Ls).to(device)
            
        if isinstance(self.model_intra_phase1.est_rho_modules, ModuleList):
            # model_intra_phase1 has various NN modules.
            
            # Repeat the update process N_step times
            for update_step in range(N_step):
                s_batch = modulus*torch.exp(1j *phi_batch)
                y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                
                x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=1)
                    
                eta_batch = self.est_eta_modules(x_batch)

                rho_batch = self.model_intra_phase1.est_rho_modules[update_step](x_batch)


                eta_stack_batch[:,update_step,:] = eta_batch
    
                # Update phi
                phi_batch = phi_batch - rho_batch*eta_batch  
    
                # save on list
                s_stack_batch[:,update_step,:] = s_batch
                
        else: # model_intra_phase1 has repeated NN module.
            
            # Repeat the update process N_step times
            for update_step in range(N_step):
                s_batch = modulus*torch.exp(1j *phi_batch)
                y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                
                x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=1)
                    
                eta_batch = self.est_eta_modules(x_batch)

                rho_batch = self.model_intra_phase1.est_rho_modules(x_batch)


                eta_stack_batch[:,update_step,:] = eta_batch
    
                # Update phi
                phi_batch = phi_batch - rho_batch*eta_batch  
    
                # save on list
                s_stack_batch[:,update_step,:] = s_batch
            
        # update along steps finished
        s_stack_batch[:,N_step,:] = modulus*torch.exp(1j *phi_batch)  # Saving the final s after all updates
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'eta_stack_batch': eta_stack_batch
        }
        return model_outputs
    
class SREL_rep_etaL(nn.Module):
    def __init__(self, constants, model_intra_phase1):
        super(SREL_rep_etaL, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        self.model_intra_phase1 = model_intra_phase1
        
        # Dynamically create the modules for estimating eta and rho
        self.est_eta_modules = Estimate_etaL(2*self.Ls + 2*constants['Lw'] + constants['Ly'], self.Ls)
        
        
    def forward(self, phi_batch, w_batch, y):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        Ls = self.Ls
        device = self.device
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, Ls, dtype=torch.complex64).to(device)
        eta_stack_batch = torch.zeros(batch_size, N_step, Ls).to(device)
            
        if isinstance(self.model_intra_phase1.est_rho_modules, ModuleList):
            # model_intra_phase1 has various NN modules.
            
            # Repeat the update process N_step times
            for update_step in range(N_step):
                s_batch = modulus*torch.exp(1j *phi_batch)
                y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                
                x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=1)
                    
                eta_batch = self.est_eta_modules(x_batch)

                rho_batch = self.model_intra_phase1.est_rho_modules[update_step](x_batch)


                eta_stack_batch[:,update_step,:] = eta_batch
    
                # Update phi
                phi_batch = phi_batch - rho_batch*eta_batch  
    
                # save on list
                s_stack_batch[:,update_step,:] = s_batch
                
        else: # model_intra_phase1 has repeated NN module.
            
            # Repeat the update process N_step times
            for update_step in range(N_step):
                s_batch = modulus*torch.exp(1j *phi_batch)
                y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                
                x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=1)
                    
                eta_batch = self.est_eta_modules(x_batch)

                rho_batch = self.model_intra_phase1.est_rho_modules(x_batch)


                eta_stack_batch[:,update_step,:] = eta_batch
    
                # Update phi
                phi_batch = phi_batch - rho_batch*eta_batch  
    
                # save on list
                s_stack_batch[:,update_step,:] = s_batch
            
        # update along steps finished
        s_stack_batch[:,N_step,:] = modulus*torch.exp(1j *phi_batch)  # Saving the final s after all updates
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'eta_stack_batch': eta_stack_batch
        }
        return model_outputs
    