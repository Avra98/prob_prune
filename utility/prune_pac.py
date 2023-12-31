import copy
import torch
import torch.nn as nn 
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def gen_output(model, prior, mask, dataset, n, criterion):
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
                param.data += torch.randn(param.data.size(), device=device)*mask[k]*prior


            errors = []
            for batch in train:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model1(inputs)
                err = criterion(predictions,targets)
                errors.extend(list(err.cpu().numpy()))

            error_list.append(errors)
            error_mean_list.append(np.mean(errors))
    return error_list, error_mean_list

def compute_K_sample(model, mask, dataset, criterion, min_gamma, max_gamma,
                        min_nu=-6, max_nu=-2.5):
    reduction = criterion.reduction
    criterion.reduction = 'none'
    def est_K(prior, x):
        # estimate k within a certain gamma range given prior
        gamma_grid = np.exp(np.linspace(np.log(min_gamma), np.log(max_gamma), 10))
        print('searching for K4....')
        error_list, error_mean_list = gen_output(model, prior, mask, dataset, 10, criterion)
            
        while min(func_sum(x, gamma_grid, error_list, error_mean_list)) < 0:
            x = x*1.1
        return x

    prior_list = np.linspace(min_nu, max_nu, 8)
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
    while x > exp_prior_list[i]:
        i +=1
        if i >= n-1:
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

def weight_decay_mulb(model, b, model_init, mask):
    # noise injection
    weights = 0
    for i, (param, param_init) in enumerate( zip(model.parameters(), model_init.parameters()) ):
        weights += torch.norm( ((param - param_init)*mask[i]).view(-1) )**2*torch.exp(-2*b[i].double())
    return weights

def get_kl_term_with_b(weight_decay, p, b):
    d = len(p)
    KL = (torch.exp(-2*b.double())*torch.exp( 2*(p).double() ).sum() /d - 
                        ( 2*(p).double().sum()/d - 2*b.double() + 1 ))
    return (KL * d + weight_decay*torch.exp(-2*b))/2

def get_kl_term_layer_pb(model, wdecay_mulb, p, b):
    k, KL1, KL2 = 0, 0, 0
    for i, param in enumerate(model.parameters()):
        t = len(param.view(-1))
        KL1 += torch.exp(-2*b[i].double())*torch.exp(2*(p[k:(k+t)]).double()).sum()
        KL2 += 2*b[i].double()*t
        k += t

    KL = KL1 - ( 2*(p).double().sum() - KL2 + len(p) )
    return (KL + wdecay_mulb)/2

######################################################################

def initialization_pac(model, mask, w0decay=1.0):
    for param in model.parameters():
        param.data *= w0decay

    device = next(model.parameters()).device
    
    w0, num_params = [], 0
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone()*mask[layer].view(-1))
        num_params += mask[layer].sum()

    num_layer = layer + 1
    w0 = torch.cat(w0) 
    p = nn.Parameter(torch.where(w0 == 0, torch.zeros_like(w0), torch.log(w0.abs().sum()/num_params)), requires_grad=True)
    prior = nn.Parameter(torch.ones(num_layer, device=device)*(torch.log(w0.abs().sum()/num_params)), requires_grad=True)
    return w0, p, num_layer, prior

def prune_by_noise_trainable_prior(model, model_init, mask, percent, train_loader_raw, criterion,
                                lr=1e-3, num_steps=1, p_init=None):

    min_gamma = 0.5
    max_gamma = 10
    min_nu=-6
    max_nu=-2.5

    train_loader = torch.utils.data.DataLoader(train_loader_raw.dataset, batch_size=1024)
    prior_list, K_list = compute_K_sample(model, mask, train_loader, criterion, min_gamma, max_gamma,
                                            min_nu, max_nu)
    print("prior:", prior_list)
    print("K:", K_list)

    device = next(model.parameters()).device
    _,p,_ ,prior= initialization_pac(model, mask)
    if p_init is not None:
        p = p_init.detach().clone()
        p.requires_grad_(True)
    p_schedule = None

    optimizer_p = torch.optim.Adam([p, prior], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer_p, mode='min', factor=0.1, patience=10)

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

            wdecay = weight_decay_mulb(model_copy, prior, model_init, mask)
            ## no noise added at the pruned locations
            k = 0
            for i, param in enumerate(model_copy.parameters()):   
                t = len(param.view(-1))
                eps = torch.randn_like(param.data, device = device)                
                noise = torch.reshape(torch.exp(p[k:(k+t)]), param.data.size()) * eps  * mask[i]                   
                param.add_(noise)  
                # mask p
                with torch.no_grad():  
                    p.data[k:(k+t)] *= mask[i].view(-1)
                k += t

            kl = get_kl_term_layer_pb(model_copy, wdecay, p, prior)
            K = fun_K_auto(torch.exp(prior.mean()),prior_list,K_list)

            gamma1 = K**(-1)*( 2*(kl+60+len(prior)*3) /len(train_loader.dataset)/3 )**0.5
            gamma1 = torch.clip(gamma1, max=max_gamma, min=min_gamma)
            kl_loss = 3*K**2*gamma1/2 + (kl+60+len(prior)*3)/len(train_loader.dataset)/gamma1

            # Forward pass after adding noise
            output = model_copy(data)
            batch_original_loss_after_noise = criterion(output, target)

            total_loss = batch_original_loss_after_noise + kl_loss
            total_loss.backward()
            optimizer_p.step()

            total_loss_accum += total_loss.item()
            kl_loss_accum += kl_loss.item()
            batch_original_loss_after_noise_accum += batch_original_loss_after_noise.item()

        # early stopping
        scheduler.step(total_loss_accum)
        # no need to keep training
        if optimizer_p.param_groups[0]['lr'] < 1e-5:
            break
        if optimizer_p.param_groups[0]['lr'] < lr - 1e-6 and p_schedule is None:
            p_schedule = p.detach().clone()

        # Average losses for the mini-batch
        print(f"Epoch {epoch+1}")
        print(f"Average batch original loss after noise: {batch_original_loss_after_noise_accum / len(train_loader):.6f}")
        print(f"Average KL loss: {kl_loss_accum / len(train_loader):.6f}")
        print(f"Average total loss: {total_loss_accum / len(train_loader):.6f}")

    k=0
    # Flatten all weights into a single list
    importance_score = []
    for i, param in enumerate(model_copy.parameters()):   
        t = param.numel()
        normalized_tensor = param.data.abs() / torch.reshape(torch.exp(p[k:(k+t)]), param.data.shape)
        importance_score.extend(normalized_tensor.flatten())
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

    if p_schedule is not None:
        return mask, p.detach().clone(), p_schedule
    else:
        return mask, p.detach().clone(), p.detach().clone()
