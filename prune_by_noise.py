# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle

def prune_by_noise(percent,train_loader,criterion, noise_type ,prior_sigma=1.0, lr=1e-3, num_steps=70):
    global model
    global mask
    global step 
    EPS=1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kl_loss = torch.zeros(1, device=device)
    torch.manual_seed(0)

    numel = sum(param.numel() for param in model.parameters())

    _,p,_ , prior_sigma, prior= initialization(model,noise_type)

    optimizer_p = torch.optim.Adam([p], lr=1e-2)

    
    
    for epoch in range(num_steps):
        # Initialize accumulators
        batch_original_loss_after_noise_accum = 0.0
        total_loss_accum = 0.0
        kl_loss_accum = 0.0

        # Loop over mini-batches
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer_p.zero_grad()
            model_copy = copy.deepcopy(model)

            for param in model_copy.parameters():
                param.requires_grad = False
                print(param)


            ## no noise added at the pruned locations
            if noise_type=="gaussian":
                k=0
                step=0
                for i, param in enumerate(model_copy.parameters()):
                    with torch.no_grad():
                        mask_torch = torch.tensor(mask[step]).to(device)

                    t = len(param.view(-1))
                    eps = torch.randn_like(param.data).to(device)                
                    noise = torch.reshape(torch.exp(p[k:(k+t)]),param.data.size()) * eps  * mask_torch
                    #with torch.no_grad(): 
                    #    if batch_idx==0:
                    #        print(torch.mean(torch.abs(param.data).view(-1)/torch.exp(p[k:(k+t)])))                                         
                    k += t  
                    step +=1                   
                    param.add_(noise) 
                step =0    
                k=0 

                kl_loss = 0.5 * torch.sum(2*prior - 2*p + (torch.exp(2*p - 2*prior) - 1))

            elif noise_type=="bernoulli":                     
                k=0
                step=0
                for i, param in enumerate(model_copy.parameters()):
                    with torch.no_grad():
                        mask_torch = torch.tensor(mask[step]).to(device)
                    t = len(param.view(-1))
                    logits = torch.reshape(p[k:(k+t)], param.data.size()).to(device)
                    #print(mask_torch.shape,logits.shape,torch.sigmoid(logits).shape)
                    noise = generate_noise_soft(torch.sigmoid(logits),temp=0.2) *mask_torch
                    with torch.no_grad():
                        if batch_idx == 0:
                            #print(f"This is p[{k}:{k+t}]:", p[k:k+t])
                            #print("This is noise:", noise)
                            ## priint mean of  p[{k}:{k+t}]:", p[k:k+t]
                            print("This is mean of noise:", torch.mean(noise))
                            print(f"This is mean of p[{k}:{k+t}]:", torch.mean(p[k:k+t]))
                        
            
                    k += t
                    step +=1
                    param.mul_(noise)

                kl_loss = (torch.sigmoid(p) * torch.log(torch.sigmoid(p)/prior) + (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p))/(1-prior))).sum()
     
            # # Forward pass after adding noise
            output = model_copy(data)
            batch_original_loss_after_noise = criterion(output, target)


            
            total_loss =  batch_original_loss_after_noise #+ 1e-3* kl_loss

            total_loss.backward()
            #Freezing noise gradients for pruned weights indieces
            # with torch.no_grad():
            #     k=0
            #     step=0
            #     for name, q in model_copy.named_parameters():
            #         #if 'weight' in name:
            #         t = len(q.view(-1))
            #         grad_tensor_p = torch.reshape(p.grad.data[k:(k+t)], mask[step].shape).cpu().numpy() * mask[step]
            #         p.grad.data[k:(k+t)] = torch.from_numpy(grad_tensor_p.flatten()).to(device)
            #         k += t
            #         step+=1
            #     k=0  
            #     step=0  

            #print(p.grad)

            optimizer_p.step()


            with torch.no_grad():
                total_loss_accum += total_loss.item()
                kl_loss_accum += kl_loss.item()
                batch_original_loss_after_noise_accum += batch_original_loss_after_noise.item()


        # Average losses for the mini-batch
        print(f"Epoch {epoch+1}")
        print(f"Average batch original loss after noise: {batch_original_loss_after_noise_accum / len(train_loader):.6f}")
        print(f"Average KL loss: {kl_loss_accum / len(train_loader):.6f}")
        print(f"Average total loss: {total_loss_accum / len(train_loader):.6f}")

        #print(f'p optimization step, batch original loss after noise: {batch_original_loss_after_noise_avg:.6f}, Total Loss: {total_loss_avg:.6f}, KL Loss: {kl_loss_avg:.6f}')
    if noise_type=="gaussian":
        with torch.no_grad():    
            k=0
            step=0


            # Flatten all weights into a single list
            all_normalized_tensors = []
            for i, (name, param) in enumerate(model_copy.named_parameters()):
                t = len(param.view(-1))
                tensor = param.data.cpu().numpy()
                normalized_tensor = np.abs(tensor) /  torch.reshape(torch.exp(p[k:(k+t)]), tensor.shape).cpu().detach().numpy()
                alive = normalized_tensor[np.nonzero(normalized_tensor)]
                all_normalized_tensors.extend(alive)
                k += t
            # Get the percentile value from all weights (as opposed to only layerwise)
            percentile_value = np.percentile(all_normalized_tensors, percent)

            # Now prune the weights
            k = 0
            step = 0
            for i, (name, param) in enumerate(model_copy.named_parameters()):
                t = len(param.view(-1))
                tensor = param.data.cpu().numpy()
                normalized_tensor = np.abs(tensor) /  torch.reshape(torch.exp(p[k:(k+t)]), tensor.shape).cpu().detach().numpy()
                weight_dev = param.device
                new_mask = np.where(normalized_tensor < percentile_value, 0, mask[step])
                        
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                k += t
                step += 1
            step = 0
            k=0

    elif noise_type=="bernoulli":
        with torch.no_grad():    
            # Apply sigmoid to the p values and convert to numpy
            p_values = p.cpu().numpy()
            # Get the percentile value from all p values
            percentile_value = np.percentile(p_values, percent)

            # Pruning the weights
            k = 0
            step = 0
            for i, (name, param) in enumerate(model_copy.named_parameters()):
                t = len(param.view(-1))
                tensor = param.data.cpu().numpy()
                weight_dev = param.device
                new_mask = np.where(torch.reshape(p_values[k:(k+t)], param.shape) < percentile_value, 0, mask[step])  # Prune based on reshaped p_values
                        
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                k += t
                step += 1



def initialization(model,noise_type ="gaussian", w0decay=1.0):
    for param in model.parameters():
        param.data *= w0decay

    device = next(model.parameters()).device
    w0 = []
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone())
    num_layer = layer + 1
    w0 = torch.cat(w0) 
    #p  = nn.Parameter(torch.ones(len(w0), device=device)*torch.log(w0.abs().mean()), requires_grad=True)
    #p  = nn.Parameter(torch.log(w0.abs()), requires_grad=True)
    if noise_type=="gaussian":
        p = nn.Parameter(torch.where(w0 == 0, torch.zeros_like(w0), torch.log(torch.abs(w0))), requires_grad=True)
        #p.data[0:int((p.numel()-1)/2)] = p.data[0:int((p.numel()-1)/2)]*2
        prior_sigma = torch.log(w0.abs().mean())
        #prior = torch.log(w0.abs())
        prior = torch.where(w0 == 0, torch.zeros_like(w0), torch.log(torch.abs(w0)))
    elif noise_type=="bernoulli":
        p = nn.Parameter(torch.zeros_like(w0), requires_grad=True)
        prior = torch.sigmoid(4*torch.ones_like(w0))
        prior_sigma = 0.0
        
        ## 1/1+e(-100)

    return w0, p, num_layer, prior_sigma,prior