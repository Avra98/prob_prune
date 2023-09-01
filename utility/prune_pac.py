import copy
import torch
import torch.nn as nn 
import numpy as np
######################################################################
# Copy from Auto-Tune
######################################################################

def func_sum(x, gamma, error_list, error_mean_list):
    def func(err, err_mu):
        out = np.zeros((len(gamma),1))
        for r in range(len(gamma)):
           out[r]= -(np.mean(np.exp( np.longdouble(gamma[r]*(err_mu-err)) ))
                        -np.exp( np.longdouble(3*(gamma[r])**2*(x**2)/2) ))
        return out

    sum_output = 0
    for i in range(len(error_mean_list)):
        sum_output += func(error_list[i],np.mean(error_mean_list))
    return sum_output


def gen_output(model, prior, mask, dataset, n, criterion, noise='gaussian'):
    error_list = []
    error_mean_list = []

    device = next(model.parameters()).device
    train = torch.utils.data.DataLoader(dataset.dataset, batch_size=1000)
    # compute the output of the random model and store it in an array
    with torch.no_grad():
        for i in range(n):
            model1 = copy.deepcopy(model)
            # generating a random model/network from the prior distribtuion
            for k, param in enumerate(model1.parameters()):
                param.data *= mask[k]
                if noise == 'gaussian':
                    param.data += torch.randn(param.data.size(), device=device)*mask[k]*prior
                else:
                    param.data = nn.functional.dropout(param.data, p = prior, training=True)

            errors = []
            for batch in train:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model1(inputs)
                err = criterion(predictions,targets)
                errors.extend(list(err.cpu().numpy()))

            error_list.append(errors)
            error_mean_list.append(np.mean(errors))
    return error_list, error_mean_list

def compute_K_sample(model, mask, dataset, criterion, min_gamma, max_gamma, noise='gaussian',
                        min_nu=-6, max_nu=-2.5):
    reduction = criterion.reduction
    criterion.reduction = 'none'
    def est_K(prior, x):
        # estimate k within a certain gamma range given prior
        gamma_grid = np.exp(np.linspace(np.log(min_gamma), np.log(max_gamma), 10))
        print('searching for K4....')
        error_list, error_mean_list = gen_output(model, prior, mask, dataset, 10, criterion, noise)
            
        while min(func_sum(x, gamma_grid, error_list, error_mean_list)) < 0:
            x = x*1.1
        return x

    prior_list = np.linspace(min_nu, max_nu, 8)
    if noise == 'gaussian':
        prior_list = np.exp(prior_list)

    K_list = [1e-3]
    for i in range(len(prior_list)):
        K_list.append(est_K(prior_list[i], K_list[-1]))
    K_list = K_list[1:]

    # make lists monotonically increasing 
    ks, priors = [], [] 
    cur_max_k = 0
    for k, p in zip(K_list, prior_list):
        if k < cur_max_k:
            ks.append(cur_max_k)
            priors.append(p)
        else:
            ks.append(k)
            priors.append(p)
            cur_max_k = k
    criterion.reduction = reduction
    return priors, ks

def fun_K_auto(x,exp_prior_list,K_list):
    n = len(exp_prior_list)
    i = 0
    while x>exp_prior_list[i]:
        i +=1
        if i == n-1:
            break
    if i==0:
        fa = K_list[0]+exp_prior_list[0]
        fb = K_list[0]
        a = 0
        b = exp_prior_list[0]
    else:
        fa = K_list[i-1]
        fb = K_list[i]
        a = exp_prior_list[i-1]
        b = exp_prior_list[i]
    return (b-x)/(b-a)*fa + (x-a)/(b-a)*fb

######################################################################
######################################################################

def generate_noise_soft(logits,temp=0.5):
    gumbel1 = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)
    gumbel2 = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)    
    numerator = torch.exp((logits + gumbel1)/temp)
    denominator = torch.exp((logits + gumbel1)/temp)  + torch.exp(((1 - logits) + gumbel2)/temp)    
    noise = numerator / denominator
    return noise

def initialization_pac(model, mask, noise_type ="gaussian", w0decay=1.0):
    for param in model.parameters():
        param.data *= w0decay

    device = next(model.parameters()).device
    
    w0, num_params = [], 0
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone()*mask[layer].view(-1))
        num_params += mask[layer].sum()

    num_layer = layer + 1
    w0 = torch.cat(w0) 
    if noise_type=="gaussian":
        p = nn.Parameter(torch.where(w0 == 0, torch.zeros_like(w0), torch.log(w0.abs())), requires_grad=True)
        prior = nn.Parameter(torch.ones(num_layer, device=device)*(torch.log(w0.abs().sum()/num_params)), requires_grad=True)
    elif noise_type=="bernoulli":
        p = nn.Parameter(torch.zeros_like(w0), requires_grad=True)
        prior = nn.Parameter(torch.zeros(num_layer, device=device), requires_grad=True)
    return w0, p, num_layer, prior

def prune_by_noise_trainable_prior(model, mask, percent,train_loader,criterion, noise_type,
                                lr=1e-3, num_steps=1):

    min_gamma = 0.5
    max_gamma = 10
    if noise_type == 'gaussian':
        min_nu=-6
        max_nu=-2.5
    else:
        min_nu=0.0
        max_nu=0.999

    prior_list, K_list = compute_K_sample(model, mask, train_loader, criterion, min_gamma, max_gamma, noise_type,
                                            min_nu, max_nu)
    print("prior:", prior_list)
    print("K:", K_list)

    device = next(model.parameters()).device
    _, p,_ ,prior= initialization_pac(model, mask, noise_type)

    optimizer_p = torch.optim.Adam([p, prior], lr=lr)

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
                k = 0
                KL = 0
                for i, param in enumerate(model_copy.parameters()):   
                    t = len(param.view(-1))
                    eps = torch.randn_like(param.data, device = device)                
                    noise = torch.reshape(torch.exp(p[k:(k+t)]), param.data.size()) * eps  * mask[i]                   
                    param.add_(noise)  
                    # mask p
                    with torch.no_grad():  
                        p.data[k:(k+t)] *= mask[i].view(-1)
                    KL += ( (2*(prior[i]-p[k:(k+t)]) + torch.exp(2*p[k:(k+t)]-2*prior[i]))*mask[i].view(-1) ).sum() - mask[i].sum() 
                    k += t

                KL *= 0.5
                gamma1 = fun_K_auto(torch.exp(prior.mean()), prior_list, K_list)**(-1)*( 2*(KL+60) /len(train_loader.dataset)/3 )**0.5
                gamma1 = torch.clip(gamma1,max=max_gamma,min=min_gamma)
                kl_loss = 3*fun_K_auto(torch.exp(prior.mean()), prior_list, K_list)**2*gamma1/2 + (KL+60)/len(train_loader.dataset)/gamma1

            elif noise_type.lower()=="bernoulli":                     
                k, KL = 0, 0
                for i, param in enumerate(model_copy.parameters()):   
                    t = len(param.view(-1))
                    logits = torch.reshape(p[k:(k+t)], param.data.size())
                    noise = generate_noise_soft(torch.sigmoid(logits),temp=0.2) * mask[i]
                    param.mul_(noise)

                    prob_p     = torch.sigmoid(p[k:(k+t)]).clamp(1e-6, 1.0)
                    prob_prior = torch.sigmoid(prior[i]).clamp(1e-6, 1.0)

                    KL += (mask[i].view(-1)*
                        (
                            prob_p * torch.log( prob_p/prob_prior ) + \
                            (1-prob_p) * torch.log( (1-prob_p)/(1-prob_prior) ) 
                        )).sum()
                    k += t
                    
                    gamma1 = fun_K_auto(prob_prior.mean(), prior_list, K_list)**(-1)*( 2*(KL+60) /len(train_loader.dataset)/3 )**0.5
                    gamma1 = torch.clip(gamma1,max=max_gamma,min=min_gamma)
                    kl_loss = 3*fun_K_auto(prob_prior.mean(), prior_list, K_list)**2*gamma1/2 + (KL+60)/len(train_loader.dataset)/gamma1

            # Forward pass after adding noise
            output = model_copy(data)
            batch_original_loss_after_noise = criterion(output, target)

            total_loss = batch_original_loss_after_noise + kl_loss
            total_loss.backward()

            # warmup steps for p
            if epoch < int(num_steps//2):
                k = 0
                for i, param in enumerate(model.parameters()):
                    t = param.data.numel()
                    num_para = mask[i].sum() 
                    num_para = 1 if num_para < 1 else num_para
                    p.grad[k:(k+t)] = p.grad[k:(k+t)].sum()/(num_para)*(torch.ones(t, device=p.device))
                    k += t

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
            #print(f" Percentile value: {percentile_value}")

            # Pruning the weights
            k = 0
            for i, param in enumerate(model_copy.parameters()):   
                t = len(param.data.view(-1))
                
                # Apply new weight and mask
                mask[i] = torch.where(torch.reshape(p[k:(k+t)], param.shape) < percentile_value, 0, mask[i])  # Prune based on reshaped p_values
                param.data = param.data * mask[i]
                k += t

    return 