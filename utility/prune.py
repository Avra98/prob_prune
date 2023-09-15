import copy
import torch
import torch.nn as nn 
import numpy as np 

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def generate_noise_soft(logits,temp=0.5):
    gumbel1 = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)
    gumbel2 = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)    
    numerator = torch.exp((logits + gumbel1)/temp)
    denominator = torch.exp((logits + gumbel1)/temp)  + torch.exp(((1 - logits) + gumbel2)/temp)    
    noise = numerator / denominator
    return noise

def initialization(model,mask,prior_sigma,noise_type ="gaussian"):
    device = next(model.parameters()).device
    
    w0, num_params = [], 0
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone()*mask[layer].view(-1))
        num_params += mask[layer].sum()

    num_layer = layer + 1
    w0 = torch.cat(w0) 
    if noise_type=="gaussian":
        p = nn.Parameter(torch.where(w0 == 0, torch.zeros_like(w0), torch.log( w0.abs().sum()/num_params) ), requires_grad=True)
        prior = torch.where(w0 == 0, torch.zeros_like(w0), torch.log( w0.abs().sum()/num_params ))
    elif noise_type=="bernoulli":
        p = nn.Parameter(torch.zeros_like(w0), requires_grad=True)
        prior = sigmoid(prior_sigma)
    return w0, p, num_layer,prior

# Prune by Percentile module
def prune_by_percentile(model, mask, percent):
    # Calculate percentile value
    importance_score = []
    for name, param in model.named_parameters():
        # We prune bias term
        #alive = torch.nonzero(param.data.abs(), as_tuple=True) # flattened array of nonzero values
        importance_score.extend(param.data.abs().view(-1))

    importance_score = torch.stack(importance_score)
    # Identify and shuffle indices of the flattened weights
    all_masks = torch.cat([m.view(-1) for m in mask])
    weight_indices = torch.arange(len(all_masks), device=all_masks.device)[all_masks > 0]
    permuted_indices = torch.argsort(importance_score[all_masks > 0])
    # Get the percentile value from all weights (as opposed to only layerwise)
    percentile_value = np.quantile(importance_score[all_masks > 0].cpu().numpy(), percent)
    # print the percentile value
    print(f'Pruning with threshold : {percentile_value}')

    num_to_prune = int(all_masks.sum() * percent)
    indices_to_prune = permuted_indices[:num_to_prune]
    all_masks[weight_indices[indices_to_prune]] = 0.0

    # Updating original weights with pruned values
    start_idx = 0
    for i, param in enumerate(model.parameters()):
        end_idx = start_idx + param.numel()
        mask[i] = all_masks[start_idx:end_idx].view(mask[i].shape)
        param.data *= mask[i] 
        start_idx = end_idx

    return mask

def prune_by_noise(model, mask, percent,train_loader_raw,criterion, noise_type ,prior_sigma=1.0, 
                        kl=0.0, lr=1e-3, num_steps=1, p_init=None):
    kl_loss = 0.0
    device = next(model.parameters()).device
    
    _,p,_ ,prior= initialization(model,mask,prior_sigma,noise_type)
    if p_init is not None:
        p = p_init.detach().clone()
        p.requires_grad_(True)

    train_loader = torch.utils.data.DataLoader(train_loader_raw.dataset, batch_size=1024)
    optimizer_p = torch.optim.Adam([p], lr=lr)

    torlence_iter, best_loss = 0, 1000000.0
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

            ## no noise added at the pruned locations
            if noise_type.lower()=="gaussian":
                k, num_params = 0, 0
                for i, param in enumerate(model_copy.parameters()):   
                    t = len(param.view(-1))
                    eps = torch.randn_like(param.data, device = device)                
                    noise = torch.reshape(torch.exp(p[k:(k+t)]),param.data.size()) * eps  * mask[i]                       
                    param.add_(noise)    
                    k += t 
                if kl:
                    kl_loss = 0.5 *(torch.sum( 2*prior - 2*p + (torch.exp(2*p - 2*prior))-1 ) ) #- num_params)

            elif noise_type.lower()=="bernoulli":                     
                k, kl_loss = 0, 0
                for i, param in enumerate(model_copy.parameters()):   
                    t = len(param.view(-1))
                    logits = torch.reshape(p[k:(k+t)], param.data.size())
                    noise = generate_noise_soft(torch.sigmoid(logits),temp=0.2) * mask[i]
                    param.mul_(noise)
                    k += t
                if kl:
                    kl_loss = (torch.sigmoid(p) * torch.log((torch.sigmoid(p)+1e-6)/prior) + (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p)+1e-6)/(1-prior))).sum()

            # Forward pass after adding noise
            output = model_copy(data)
            batch_original_loss_after_noise = criterion(output, target)

            if kl:
                total_loss = batch_original_loss_after_noise + kl*kl_loss
            else:                    
                total_loss =  batch_original_loss_after_noise
            total_loss.backward()
            
            with torch.no_grad():
                if batch_idx==0:
                    print(torch.mean(p),torch.var(p),torch.mean(p.grad))
            optimizer_p.step()


            with torch.no_grad():
                total_loss_accum += total_loss.item()
                if kl:
                    kl_loss_accum += kl_loss.item()
                batch_original_loss_after_noise_accum += batch_original_loss_after_noise.item()

            if total_loss_accum / len(train_loader) <= best_loss:
                torlence_iter = 0
                best_loss = total_loss_accum / len(train_loader)
            else:
                torlence_iter += 1

            if torlence_iter > 10:
                break


        # Average losses for the mini-batch
        print(f"Epoch {epoch+1}")
        print(f"Average batch original loss after noise: {batch_original_loss_after_noise_accum / len(train_loader):.6f}")
        if kl:
            print(f"Average KL loss: {kl_loss_accum / len(train_loader):.6f}")
        print(f"Average total loss: {total_loss_accum / len(train_loader):.6f}")

    if noise_type.lower()=="gaussian":
        with torch.no_grad():    
            k=0
            # Flatten all weights into a single list
            importance_score = []
            for i, param in enumerate(model_copy.parameters()):   
                t = param.numel()
                normalized_tensor = param.data.abs() / torch.reshape(torch.exp(p[k:(k+t)]), param.data.shape)
                importance_score.extend(normalized_tensor.flatten())
                k += t
            importance_score = torch.stack(importance_score)

    elif noise_type.lower()=="bernoulli":
        with torch.no_grad():    
            
            importance_score = []
            k = 0
            for m in mask:
                t = m.numel()  # total elements in current layer
                importance_score.extend(p[k:(k+t)])
                k += t           
            importance_score = torch.stack(importance_score)

    # Identify and shuffle indices of the flattened weights
    all_masks = torch.cat([m.view(-1) for m in mask])
    weight_indices = torch.arange(len(all_masks), device=all_masks.device)[all_masks > 0]
    permuted_indices = torch.argsort(importance_score[all_masks > 0])
    
    num_to_prune = int(all_masks.sum() * percent)
    indices_to_prune = permuted_indices[:num_to_prune]
    all_masks[weight_indices[indices_to_prune]] = 0.0

    # Get the percentile value from all weights (as opposed to only layerwise)
    percentile_value = np.quantile(importance_score[all_masks > 0].cpu().numpy(), percent)
    print(f" Percentile value: {percentile_value}")

    # Updating original weights with pruned values
    start_idx = 0
    for i, param in enumerate(model.parameters()):
        end_idx = start_idx + param.numel()
        mask[i] = all_masks[start_idx:end_idx].view(mask[i].shape)
        param.data *= mask[i] 
        start_idx = end_idx

    return mask, p.detach().clone()

def prune_by_random(model, mask, percent):
    # Flatten all weights of the model
    all_masks = torch.cat([m.view(-1) for m in mask])

    # Identify and shuffle indices of the flattened weights
    weight_indices = torch.arange(len(all_masks), device=all_masks.device)[all_masks > 0]
    permuted_indices = torch.randperm(weight_indices.size(0))
    
    # Calculate the number of weights to set to zero
    num_to_prune = int(all_masks.sum() * percent)
    indices_to_prune = permuted_indices[:num_to_prune]
    
    # Set those weights to zero
    all_masks[weight_indices[indices_to_prune]] = 0.0
    
    # Updating original weights with pruned values
    start_idx = 0
    for i, param in enumerate(model.parameters()):
        end_idx = start_idx + param.numel()
        mask[i] = all_masks[start_idx:end_idx].view(mask[i].shape)
        param.data *= mask[i] 
        start_idx = end_idx
    return mask