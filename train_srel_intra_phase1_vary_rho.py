# -*- coding: utf-8 -*-
"""
Created on April 2 2024

@author: jbk5816
"""

import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# from utils.complex_valued_dataset import ComplexValuedDataset
from utils.training_dataset import TrainingDataSet
from utils.load_scalars_from_setup import load_scalars_from_setup

from model.srel_intra_phase1 import SREL_intra_phase1_vary_rho
from model.srel_intra_tester import SREL_intra_phase1_tester
# print('SRED_rho with Batch Normalization (BN)')


from utils.custom_loss_intra import custom_loss_intra_phase1_mono

from torch.utils.tensorboard import SummaryWriter #tensorboard
# tensorboard --logdir=runs/SREL --reload_interval 5
# tensorboard --logdir=runs/SREL_intra
from visualization.plotting import plot_losses # result plot

import datetime
import time
import os
import argparse

from utils.validation import validation
from utils.save_result_mat import save_result_mat

from utils.format_time import format_time

# import torch.nn as nn

def main(save_weights, save_logs, save_mat, 
         batch_size, learning_rate, lambda_sinr, lambda_mono):
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    # y_M, Ly = load_mapVector('data/data_mapV.mat')
    data_num = 1e1
    
    
    # loading constant
    constants['Ly'] = 570
    Nt = constants['Nt']
    M = constants['M']
    N = constants['N']
    
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))
    
    # Check for GPU availability.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(0)
    ###############################################################
    ## Control Panel
    ###############################################################
    # Initialize model
    N_step = 10
    constants['N_step'] = N_step
    model_intra_phase1 = SREL_intra_phase1_vary_rho(constants)
#    model_intra_phase1.apply(init_weights)
    num_epochs = 50
    # Initialize the optimizer
    learning_rate=1e-5
    print(f'learning_rate={learning_rate:.0e}')
    optimizer = optim.Adam(model_intra_phase1.parameters(), lr=learning_rate)
    
    # loss setting
    lambda_sinr = 1e-2
    lambda_var_rho = 0
    hyperparameters = {
        'lambda_sinr': lambda_sinr,
        'lambda_var_rho': lambda_var_rho,
        'lambda_mono': lambda_mono,
    }    
    ###############################################################
    model_intra_phase1.to(device)
    model_intra_phase1.device = device
    
    # for results
    # Get the current time
    start_time_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') 
    print(f'current time: {start_time_tag}')
    
    # Create a unique directory name using the current time and the N_step value
    log_dir = (
        f'runs/intra_phase1/vary_rho/data{data_num:.0e}/{start_time_tag}'
        f'_Nstep{constants["N_step"]:02d}_batch{batch_size:02d}'
        f'_lr_{learning_rate:.0e}'
    )
    writer = SummaryWriter(log_dir)    
    
    # List to store average loss per epoch
    training_losses = []
    validation_losses = []
    
    
    start_time_total = time.time()
    
    # Training loop
    num_case = 24
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_val_loss = 0.0
        
        sum_of_worst_sinr_avg = 0.0  # Accumulate loss over all batches
        
        start_time_epoch = time.time()  # Start timing the inner loop
        for idx_case in range(num_case):
            case_num = idx_case + 1
            dataset = TrainingDataSet(f'data/{data_num:.0e}/data_trd_{data_num:.0e}_case{case_num:02d}.mat')

            # Split dataset into training and validation
            train_indices, val_indices = train_test_split(
                range(len(dataset)),
                test_size=0.2,  # 20% for validation
                random_state=42
            )
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

            # batch_size = 10
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model_intra_phase1.train()  # Set model to training mode
            
            for phi_batch, w_M_batch, G_M_batch, H_M_batch in train_loader:
                phi_batch = phi_batch.to(device)
                G_M_batch = G_M_batch.to(device)
                H_M_batch = H_M_batch.to(device)
                w_M_batch = w_M_batch.to(device)
                y_M = dataset.y_M.to(device)  # If y_M is a tensor that requires to be on the GPU

                for m, (G_batch, H_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                           torch.unbind(H_M_batch, dim=3))):
                    w_batch = w_M_batch[:,:,m]
                    y = y_M[:,m]

                    # Perform training steps
                    optimizer.zero_grad()
    
                    model_outputs = model_intra_phase1(phi_batch, w_batch, y, G_batch, H_batch)

                    # s_stack_batch = model_outputs['s_stack_batch']
                    loss, _, _ = custom_loss_intra_phase1_mono(
                        constants, G_batch, H_batch, hyperparameters, model_outputs)
    
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()



            # Validation phase
            model_intra_phase1.eval()  # Set model to evaluation mode
            model_intra_tester = SREL_intra_phase1_tester(constants, model_intra_phase1).to(device)
            model_intra_tester.device = device

            
            # sum_of_worst_sinr_avg = 0.0
            
            with torch.no_grad():  # Disable gradient computation
                for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
                    # s_batch = modulus * torch.exp(1j * phi_batch)
                    phi_batch = phi_batch.to(device)
                    G_M_batch = G_M_batch.to(device)
                    H_M_batch = H_M_batch.to(device)
                    w_M_batch = w_M_batch.to(device)
                    y_M = y_M.to(device)  # If y_M is a tensor that requires to be on the GPU
                    
                    sinr_opt_M = np.zeros(M)
                    
                    for m, (G_batch, H_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                               torch.unbind(H_M_batch, dim=3))):
                        w_batch = w_M_batch[:,:,m]
                        y = y_M[:,m]
                        
                        model_outputs = model_intra_phase1(phi_batch, w_batch, y, G_batch, H_batch)

                        val_loss, sinr_opt_avg, _ = custom_loss_intra_phase1_mono(
                            constants, G_batch, H_batch, hyperparameters, model_outputs)
                        
                        total_val_loss += val_loss.item()
                        
                        # model_intra_tester
                        
                        # s_stack_batch = model_outputs['s_stack_batch']
                        # s_optimal_batch = s_stack_batch[:,-1,:].squeeze()
                        
                        sinr_opt_M[m] = sinr_opt_avg.item()
                        
                    sum_of_worst_sinr_avg += np.min(sinr_opt_M)
    
                    # model_outputs = model_intra_tester(
                    #     phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch
                    #     )
                    
                    # s_stack_batch = model_outputs['s_stack_batch']
                    # s_optimal_batch = s_stack_batch[:,-1,:]
                    # sum_of_worst_sinr_avg += np.sum(
                    #     worst_sinr_function(
                    #         constants, s_optimal_batch, G_M_batch, H_M_batch)
                    #     )/batch_size
                    
                    
                    
        # Compute average loss for the epoch and Log the loss
        average_train_loss = total_train_loss / M / len(train_loader) / num_case
        average_train_loss_db = 10*np.log10(average_train_loss)
        training_losses.append(average_train_loss_db)
        
        average_val_loss = total_val_loss / M / len(test_loader) / num_case
        average_val_loss_db = 10*np.log10(average_val_loss)
        validation_losses.append(average_val_loss_db)
        
        if save_logs:
            writer.add_scalar('Loss/Training [dB]', average_train_loss_db, epoch)
            writer.add_scalar('Loss/Testing [dB]', average_val_loss_db, epoch)
            writer.flush()
    
        # Log the loss        
        worst_sinr_avg_db = 10*np.log10(sum_of_worst_sinr_avg/ len(test_loader) / num_case)  # Compute average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], '
             f'Train Loss = {average_train_loss_db:.2f} dB, '
             f'average_worst_sinr = {worst_sinr_avg_db:.4f} dB')
        
        time_spent_epoch = time.time() - start_time_epoch  # Time spent in the current inner loop iteration
        time_left = time_spent_epoch * (num_epochs - epoch - 1)  # Estimate the time left
        formatted_time_left = format_time(time_left)
        print(f"{formatted_time_left} left")
        
        # print(f'Epoch [{epoch+1}/{num_epochs}], '
              # f'Train Loss = {average_train_loss:.4f}, '
              # f'average_worst_sinr = {worst_sinr_avg_db:.4f} dB')
        
        
        
        
    # End time
    end_time = time.time()
    
    # Calculate the time_spent_total
    time_spent_total = end_time - start_time_total
    
    print(f"Training completed in: {time_spent_total:.2f} seconds")
    
    plot_losses(training_losses, validation_losses)
    
    # validation
    worst_sinr_stack_list, f_stack_list = validation(constants,model_intra_tester)
    # sinr_db_opt = 10*np.log10(
    #     np.mean(worst_sinr_stack_list[:,-1])
    #     )
    
    if save_mat:
        matfilename = "data_SRED_rho_10step_result.mat"
        dir_mat_save = (
            f'mat/intra_phase1/vary_rho/{start_time_tag}'
            f'_Nstep{N_step:02d}_batch{batch_size:02d}'
            f'_sinr_{worst_sinr_avg_db:.2f}dB'
        )
        os.makedirs(dir_mat_save, exist_ok=True)
        save_result_mat(os.path.join(dir_mat_save, matfilename), worst_sinr_stack_list, f_stack_list)
    

    # save model's information
    if save_weights:
        save_dict = {
            'state_dict': model_intra_phase1.state_dict(),
            'N_step': model_intra_phase1.N_step,
            # Include any other attributes here
        }
        # save
        dir_weight_save = (
            f'weights/intra_phase1/vary_rho/{start_time_tag}'
            f'_Nstep{N_step:02d}_batch{batch_size:02d}'
            f'_sinr_{worst_sinr_avg_db:.2f}dB'
        )
        os.makedirs(dir_weight_save, exist_ok=True)
        torch.save(save_dict, os.path.join(dir_weight_save, 'model_with_attrs.pth'))
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    
    parser.add_argument("--save-weights", action="store_true",
                        help="Save the model weights after training")
    parser.add_argument("--save-logs", action="store_true",
                        help="Save logs for Tensorboard after training")
    parser.add_argument("--save-mat", action="store_true",
                        help="Save mat file including worst-sinr values")
    
    args = parser.parse_args()
    
    main(save_weights=args.save_weights, save_logs=args.save_logs,save_mat=args.save_mat, 
        batch_size=5, learning_rate=1e-5, lambda_sinr = 1e-4, lambda_mono=1e-1)
    
    # main(save_weights=args.save_weights, save_logs=args.save_logs,save_mat=args.save_mat, 
    #     batch_size=5, learning_rate=1e-5, lambda_sinr = 1e-4, lambda_mono=1e-3)
    
    # main(save_weights=args.save_weights, save_logs=args.save_logs,save_mat=args.save_mat, 
    #     batch_size=5, learning_rate=1e-5, lambda_sinr = 1e-4, lambda_mono=1e-5)

