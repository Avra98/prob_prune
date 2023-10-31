import copy
import torch
import torch.nn as nn 
import numpy as np 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

import torch.nn.functional as F


def inverse_sigmoid(x):
    return torch.log(torch.tensor(x) / torch.tensor(1 - x))

# def gumbel_softmax_multi(logits, temperature=0.2):
#     # print("logits shape is before: ", logits.shape)

#     #num_classes = logits.shape[1]
    
#     # Add a column for 1-p1-p2
#     logits = torch.cat((logits, 1 - logits.sum(dim=1, keepdim=True)), dim=1) 
#     # print("logits shape is after: ", logits.shape)  
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)
#     y = (torch.log(logits) + gumbel_noise) / temperature
#     softmax_output = F.softmax(y, dim=-1)   
#     return softmax_output


def gumbel_softmax_multi(logits, temperature=0.2):
    # Add a column for 1-p1-p2, clamped to prevent log(0)
    logits = torch.cat((logits, 1 - logits.sum(dim=1, keepdim=True).clamp(min=1e-20, max=1-1e-20)), dim=1) 
    
    # Gumbel noise; explicitly without gradients, clamped to prevent log(0)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-20, max=1-1e-20))).requires_grad_(False)
    
    # Logits; clamped to prevent log(0)
    y = (torch.log(logits.clamp(min=1e-20, max=1-1e-20)) + gumbel_noise) / temperature
    
    # Softmax output
    softmax_output = F.softmax(y, dim=-1)
    
    return softmax_output


# def soft_quantize(logits,temperature):
#     soft_samples = gumbel_softmax_multi(logits, temperature)
#     # if batch_idx==0:
#     #     print("soft_samples is: ", torch.mean(soft_samples, dim=0))
#     quant_values = torch.tensor([1,0.5,0.0]).to(logits.device)
#     quantized_output = torch.sum(soft_samples * quant_values, dim=-1)
#     return quantized_output

def soft_quantize(logits, q, temperature=0.2):
    soft_samples = gumbel_softmax_multi(logits, temperature)
    #print("soft_samples is: ", soft_samples.shape)
    with torch.no_grad():
        quant_values = [torch.tensor([1 - i / (q - 1)]) for i in range(q - 1)] + [torch.tensor([0.0])]
        quant_values = torch.cat(quant_values).to(logits.device)
    # print("quant_values is: ", quant_values)
    quantized_output = torch.sum(soft_samples * quant_values, dim=-1)
    return quantized_output

def quant_initialization(model,mask,prior_sigma,q=3):
    device = next(model.parameters()).device
    
    w0, num_params = [], 0
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone()*mask[layer].view(-1))
        num_params += mask[layer].sum()
    num_layer = layer + 1
    w0 = torch.cat(w0) 
    #p = nn.Parameter(torch.zeros_like(w0), requires_grad=True)
    p =  nn.Parameter(inverse_sigmoid(1.5/(q))*torch.ones([q-1, w0.size(0)]).to(device), requires_grad=True)
    # p =  nn.Parameter(-5*torch.ones([q-1, w0.size(0)]).to(device), requires_grad=True)
    with torch.no_grad():  
        p[0, :].fill_((1.5/q)-0.25)
    #     print("p elementS are",p[:,0])
    prior = sigmoid(prior_sigma)
    return w0, p, num_layer,prior

def quant_by_noise(model, mask, percent, train_loader_raw, criterion, prior_sigma=1.0, 
                   kl=0.0, num_steps=1, lr=1e-3,  p_init=None, reduce_op=False, q=3):
    
    kl_loss = 0.0
    device = next(model.parameters()).device
    
    #_,p,_ ,prior= initialization(model,mask,prior_sigma,noise_type)
    _, p, _, prior = quant_initialization(model, mask, prior_sigma, q)
    if p_init is not None:
        #p = p_init.detach().clone()
        p.requires_grad_(True)
    p_schedule = None
    
    num_params = 0
    # for m in mask:
    #     num_params += m.sum()
    train_loader = torch.utils.data.DataLoader(train_loader_raw.dataset, 
                    batch_size=1024, shuffle=True, num_workers=4)
    optimizer_p = torch.optim.Adam([p], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer_p, mode='min', factor=0.1, patience=10)

    tolerance, best_loss = 0, 1000000.0
    all_quantized_weights = []

    for epoch in range(num_steps):
        # ... [Loss Accumulators]
        batch_original_loss_after_quantization_accum = 0.0
        total_loss_accum = 0.0
        kl_loss_accum = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer_p.zero_grad()
            model_copy = copy.deepcopy(model)
            for param in model_copy.parameters():
                param.requires_grad = False

            k = 0
            for i, param in enumerate(model_copy.parameters()):   
                t = len(param.view(-1))
                logits = p[:, k:(k+t)].t()
                quantized_weights = soft_quantize(torch.sigmoid(logits),q,temperature=0.2)
                with torch.no_grad():
                    if batch_idx==0:
                        #print("logits is: ", logits.mean())
                        print("quantized_weights is: ", quantized_weights)
                        if epoch == num_steps - 1:
                            quantized_weights_flat = quantized_weights.view(-1).cpu().detach().numpy()
                            all_quantized_weights.extend(quantized_weights_flat)
                #param.data = param.data * quantized_weights.view(param.data.shape)
                param.mul_(quantized_weights.view(param.data.shape))
                k += t 
    
            # if kl:
            #     kl_loss = torch.sigmoid(p).sum()

            # Forward pass after quantization
            output = model_copy(data)
            batch_original_loss_after_quantization = criterion(output, target)
            total_loss = batch_original_loss_after_quantization #+ kl*kl_loss
            #with torch.no_grad():
            total_loss_accum += total_loss.item()
            total_loss.backward()

            # ... [Loss Computations]
            with torch.no_grad():
                if batch_idx==0:
                    print(torch.mean(p),torch.var(p),torch.mean(p.grad))
            
            # ... [Backward Pass]

            optimizer_p.step()
            # if kl:               
            #     kl_loss_accum += kl_loss.item()
            batch_original_loss_after_quantization_accum += batch_original_loss_after_quantization.item()

        # early stopping
        scheduler.step(total_loss_accum)
        # no need to keep training
        if optimizer_p.param_groups[0]['lr'] < 1e-5:
            break
        if optimizer_p.param_groups[0]['lr'] < lr - 1e-6 and p_schedule is None:
            p_schedule = p.detach().clone()


       # Average losses for the mini-batch
        print(f"Epoch {epoch+1}")
        print(f"Average batch original loss after noise: {batch_original_loss_after_quantization_accum / len(train_loader):.6f}")
        # if kl:
        #     print(f"Average KL loss: {kl*kl_loss_accum / len(train_loader):.6f}")
        print(f"Average total loss: {total_loss_accum / len(train_loader):.6f}")
        # ... [Loss Reporting]

    # ... [Pruning Steps based on Quantization Importance]
    ## update the actual model based on the quantized weights
    with torch.no_grad():
        k = 0
        for i, param in enumerate(model.parameters()):
            t = len(param.view(-1))
            logits = p[:, k:(k+t)].t()
            quantized_weights = soft_quantize(torch.sigmoid(logits),q,temperature=0.2)
            param.data = param.data * quantized_weights.view(param.data.shape)
            k += t
        plt.hist(all_quantized_weights, bins=50, alpha=0.5, label='All Layers')
        plt.title(f'Quantized Weights Histogram for All Layers (q={q})')
        plt.xlabel('Quantized Weight Values')
        plt.ylabel('Frequency')

        # Save the figure with q in the filename
        fig_filename = f'all_layers_histogram_q{q}.png'
        plt.savefig(fig_filename)
        plt.clf()    


    return model, p

