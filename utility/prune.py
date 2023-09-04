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

def initialization(model,prior_sigma,noise_type ="gaussian", w0decay=1.0):
    for param in model.parameters():
        param.data *= w0decay

    device = next(model.parameters()).device
    w0 = []
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone())
    num_layer = layer + 1
    w0 = torch.cat(w0) 
    if noise_type=="gaussian":
        #p = nn.Parameter(torch.where(w0 == 0, torch.zeros_like(w0), torch.log(torch.abs(w0))), requires_grad=True)
        p = nn.Parameter(torch.where(w0 == 0, torch.zeros_like(w0), 
            torch.log(torch.mean(torch.abs(w0))*torch.ones_like(w0))), requires_grad=True)
        #prior = torch.where(w0 == 0, torch.zeros_like(w0), torch.log(torch.abs(w0)))
        prior = torch.where(w0 == 0, torch.zeros_like(w0), torch.log(torch.mean(torch.abs(w0))*torch.ones_like(w0)))
    elif noise_type=="bernoulli":
        p = nn.Parameter(torch.zeros_like(w0), requires_grad=True)
        prior = sigmoid(prior_sigma)
    return w0, p, num_layer,prior

# Prune by Percentile module
def prune_by_percentile(model, mask, percent):
    # Calculate percentile value
    all_alive_weights = []
    for name, param in model.named_parameters():
        # We prune bias term
        alive = torch.nonzero(param.data.abs(), as_tuple=True) # flattened array of nonzero values
        all_alive_weights.extend(param.data.abs()[alive])

    all_alive_weights = torch.stack(all_alive_weights)
    # Get the percentile value from all weights (as opposed to only layerwise)
    percentile_value = np.quantile(all_alive_weights.cpu().numpy(), percent)

    ##print the percentile value
    print(f'Pruning with threshold : {percentile_value}')

    # Now prune the weights
    for i, param in enumerate(model.parameters()):
        new_mask = torch.where(param.data.abs() < percentile_value, 0, mask[i])
                
        # Apply new weight and mask
        param.data = param.data * new_mask
        mask[i] = new_mask
    return

def prune_by_noise(model, mask, percent,train_loader,criterion, noise_type ,prior_sigma=1.0, 
                        kl=0.0, lr=1e-3, num_steps=1):
    kl_loss = 0.0
    device = next(model.parameters()).device
    _,p,_ ,prior= initialization(model,prior_sigma,noise_type)
    optimizer_p = torch.optim.Adam([p], lr=lr)

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
                    with torch.no_grad():
                         p.data[k:(k+t)] *= mask[i].view(-1)
                    k += t 
                    #num_params += mask[i].sum() 
                if kl:
                    kl_loss = 0.5 *(torch.sum( 2*prior - 2*p + (torch.exp(2*p - 2*prior))-1 ) ) #- num_params)

            elif noise_type.lower()=="bernoulli":                     
                k, kl_loss = 0, 0
                for i, param in enumerate(model_copy.parameters()):   
                    t = len(param.view(-1))
                    logits = torch.reshape(p[k:(k+t)], param.data.size())
                    noise = generate_noise_soft(torch.sigmoid(logits),temp=0.2) * mask[i]
                    param.mul_(noise)
                    
                    # if kl:
                    #     kl_loss += (mask[i].view(-1)*(
                    #         torch.sigmoid(p[k:(k+t)]) * torch.log((torch.sigmoid(p[k:(k+t)])+1e-6)/prior) + \
                    #         (1-torch.sigmoid(p[k:(k+t)])) * torch.log((1-torch.sigmoid(p[k:(k+t)])+1e-6)/(1-prior)))).sum()
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
            all_normalized_tensors = []
            for i, param in enumerate(model_copy.parameters()):   
                t = len(param.view(-1))
                normalized_tensor = param.data.abs() / torch.reshape(torch.exp(p[k:(k+t)]), param.data.shape)
                alive = normalized_tensor[torch.nonzero(normalized_tensor,as_tuple=True)]
                # print(alive.shape)
                all_normalized_tensors.extend(alive)
                k += t
            all_normalized_tensors = torch.stack(all_normalized_tensors)
            # print(all_normalized_tensors.shape)
            # Get the percentile value from all weights (as opposed to only layerwise)
            percentile_value = np.quantile(all_normalized_tensors.cpu().numpy(), percent)
            # Now prune the weights
            k = 0
            for i, param in enumerate(model_copy.parameters()):   
                t = len(param.view(-1))
                normalized_tensor = param.data.abs() / torch.reshape(torch.exp(p[k:(k+t)]), param.data.shape)
                # Apply new weight and mask
                mask[i] = torch.where(normalized_tensor < percentile_value, 0, mask[i])
                param.data = param.data * mask[i] 
                k += t

    elif noise_type.lower()=="bernoulli":
        with torch.no_grad():    
            
            unpruned_p_values = []
            k = 0
            for m in mask:
                t = m.numel()  # total elements in current layer
                layer_p_values = p[k:(k+t)]  # get p_values for the current layer
                unpruned_indices = torch.nonzero(m.flatten())  # unpruned indices for the current layer
                unpruned_p_values.extend(layer_p_values[unpruned_indices])
                k += t           
            unpruned_p_values = torch.stack(unpruned_p_values)
            # Get the percentile value from all p values
            percentile_value = np.quantile(unpruned_p_values.cpu().numpy(), percent)
            print(f" Percentile value: {percentile_value}")

            # Pruning the weights
            k = 0
            for i, param in enumerate(model_copy.parameters()):   
                t = len(param.data.view(-1))
                
                # Apply new weight and mask
                mask[i] = torch.where(torch.reshape(p[k:(k+t)], param.shape) < percentile_value, 0, mask[i])  # Prune based on reshaped p_values
                param.data = param.data * mask[i]
                k += t

    return 
        