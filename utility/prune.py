import copy
import torch
import torch.nn as nn 

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
        p = nn.Parameter(torch.where(w0 == 0, torch.zeros_like(w0), torch.log(torch.abs(w0))), requires_grad=True)
        prior = torch.where(w0 == 0, torch.zeros_like(w0), torch.log(torch.abs(w0)))
    elif noise_type=="bernoulli":
        p = nn.Parameter(torch.zeros_like(w0), requires_grad=True)
        prior = torch.sigmoid(prior_sigma*torch.ones_like(w0))
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
    percentile_value = torch.quantile(all_alive_weights, percent)
    ##print the percentile value
    print(f'Pruning with threshold : {percentile_value}')

    # Now prune the weights
    for i, param in enumerate(model.parameters()):
            new_mask = torch.where(param.data.abs() < percentile_value, 0, mask[i])
                    
            # Apply new weight and mask
            param.data = param.data * new_mask
            mask[i] = new_mask
    return

def prune_by_noise(model, mask, percent,train_loader,criterion, noise_type ,prior_sigma=1.0, kl=False, lr=1e-3, num_steps=1):
    #torch.manual_seed(0)
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
                k=0
                for i, param in enumerate(model_copy.parameters()):   
                    t = len(param.view(-1))
                    eps = torch.randn_like(param.data, device = device)                
                    noise = torch.reshape(torch.exp(p[k:(k+t)]),param.data.size()) * eps  * mask[i]             
                    k += t    
                    # with torch.no_grad():
                        ## print sum of noise and also batch_idx number 
                        # if batch_idx==0:
                        #     print(i,torch.sum(eps))        
                    param.add_(noise)    
                if kl:
                    kl_loss = 0.5 * torch.sum(2*prior - 2*p + (torch.exp(2*p - 2*prior) - 1))

            elif noise_type.lower()=="bernoulli":                     
                k=0
                for i, param in enumerate(model_copy.parameters()):   
                    t = len(param.view(-1))
                    logits = torch.reshape(p[k:(k+t)], param.data.size())
                    noise = generate_noise_soft(torch.sigmoid(logits),temp=0.2) * mask[i]
                    k += t
                    param.mul_(noise)
                if kl:
                    kl_loss = (torch.sigmoid(p) * torch.log((torch.sigmoid(p)+1e-6)/prior) + \
                                (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p)+1e-6)/(1-prior))).sum()
     
            # Forward pass after adding noise
            output = model_copy(data)
            batch_original_loss_after_noise = criterion(output, target)

            if kl:
                total_loss = batch_original_loss_after_noise + 1e-4*kl_loss
            else:                    
                total_loss =  batch_original_loss_after_noise
            total_loss.backward()
            with torch.no_grad():
                if batch_idx==1:    
                    print(torch.mean(p))
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
                all_normalized_tensors.extend(alive)
                k += t
            all_normalized_tensors = torch.stack(all_normalized_tensors)
            # Get the percentile value from all weights (as opposed to only layerwise)
            percentile_value = torch.quantile(all_normalized_tensors, percent)

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
            percentile_value = torch.quantile(unpruned_p_values, percent)
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


def prune_by_random(model, mask, percent):    
    # Collect all the alive weight indices and the non-zero mask indices
    alive_indices = []
    mask_alive_indices = []

    dim = 0
    for i, param in enumerate(model.parameters()):
        flat_mask = mask[i].view(-1)

        tensor_alive_indices = torch.nonzero(param.data.view(-1)) + dim
        mask_alive_indices_step = torch.nonzero(flat_mask) + dim

        # Get the indices of non-zero elements in both tensor and mask
        alive_indices.extend(tensor_alive_indices)
        mask_alive_indices.extend(mask_alive_indices_step)

        dim += param.data.numel()

    alive_indices = torch.cat(alive_indices)
    mask_alive_indices = torch.cat(mask_alive_indices)
    num_weights_to_prune = int(percent * len(alive_indices))
    
    perm = torch.randperm(len(alive_indices))
    indices_to_prune = alive_indices[perm[:num_weights_to_prune]]
    
    # Update the masks and parameters
    current_index = 0
    for i, param in enumerate(model.parameters()):
        segment = torch.where(torch.logical_and(current_index <= indices_to_prune,
                                                indices_to_prune < current_index + param.data.numel()),
                                indices_to_prune, 0)
        flat_mask = mask[i].view(-1)
        flat_mask[segment[segment>0]-current_index] = 0
        mask[i] = flat_mask.view(mask[i].shape)
        
        # Apply new weight and maskn  
        param.data = param.data * mask[i]

        current_index += param.data.numel()
        