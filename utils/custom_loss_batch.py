# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:23:14 2024

@author: jbk5816
"""

import torch
# from utils.functions import eta_sred

def reciprocal_sinr(G_batch, H_batch, s_batch):
    s_batch_unsqueezed = s_batch.unsqueeze(-1)
    
    
    # I suspect this part might not be correct
    Gs_batch = torch.bmm(G_batch, s_batch_unsqueezed).squeeze()
    Hs_batch = torch.bmm(H_batch, s_batch_unsqueezed).squeeze()
    
    sGs_batch = torch.abs(torch.sum(torch.conj(s_batch) * Gs_batch, dim=1)).unsqueeze(-1)
    sHs_batch = torch.abs(torch.sum(torch.conj(s_batch) * Hs_batch, dim=1)).unsqueeze(-1)
    
    f_sinr_m = sGs_batch / sHs_batch
     
    return f_sinr_m.squeeze()




# def fidelity_eta(G_batch, H_batch, s_batch):
#     eta_batch = eta_sred(G_batch, H_batch, s_batch)
    
    
    