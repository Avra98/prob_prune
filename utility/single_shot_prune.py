import torch
import numpy as np

def prune_by_mag(model, percent, dataloader, device, loss_fn):
    # Compute gradient for each parameter
    # data, target = next(iter(dataloader))
    # data, target = data.to(device), target.to(device)
    # output = model(data)
    # loss = loss_fn(output, target)
    # loss.backward()

    # Calculate SNIP scores |w * grad|
    scores = {}
    for name, param in model.named_parameters():
        scores[name] = torch.abs(param.data)
        param.grad = None  # clear gradient

    # Get the percentile threshold
    all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
    threshold = np.quantile(all_scores.cpu().numpy(), percent)

    # Prune parameters below the threshold
    for name, param in model.named_parameters():
        mask = torch.where(scores[name] < threshold, torch.tensor(0., device=device), torch.tensor(1., device=device))
        param.data *= mask


def prune_by_snip(model, percent, dataloader, device, loss_fn):
    # Compute gradient for each parameter
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()

    # Calculate SNIP scores |w * grad|
    scores = {}
    for name, param in model.named_parameters():
        scores[name] = torch.abs(param.data * param.grad)
        param.grad = None  # clear gradient

    # Get the percentile threshold
    all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
    threshold = np.quantile(all_scores.cpu().numpy(), percent)

    # Prune parameters below the threshold
    for name, param in model.named_parameters():
        mask = torch.where(scores[name] < threshold, torch.tensor(0., device=device), torch.tensor(1., device=device))
        param.data *= mask

def prune_by_grasp(model, percent, dataloader, device, loss_fn):
    # Compute gradient for each parameter
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()

    grads = {name: param.grad for name, param in model.named_parameters()}
    
    # Compute Hessian-vector product: \grad(|| \grad L||^2)
    hessian_vector_product = torch.autograd.grad(torch.sum(torch.pow(loss, 2)), model.parameters(), retain_graph=True)

    # Calculate GRASP scores |w * Hessian-vector|
    scores = {}
    for (name, param), hv in zip(model.named_parameters(), hessian_vector_product):
        scores[name] = torch.abs(param.data * hv)
        param.grad = None  # clear gradient

    # Get the percentile threshold
    all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
    threshold = np.quantile(all_scores.cpu().numpy(), percent)

    # Prune parameters below the threshold
    for name, param in model.named_parameters():
        mask = torch.where(scores[name] < threshold, torch.tensor(0., device=device), torch.tensor(1., device=device))
        param.data *= mask