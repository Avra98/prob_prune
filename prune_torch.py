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
from sam import SAM 

# Custom Libraries
import utils

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

# Main
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reinit = True if args.prune_type=="reinit" else False
    ## Confirm why gpu utility is so low 
    # GPU Utility

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet,fcs

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet ,fcs

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  ,fcs
    
    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    #train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)
    
    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)   
    elif args.arch_type == "fcs":
        model = fcs.fcs().to(device)    
    # If you want to add extra model paste here
    else:
        print("\nWrong Model choice\n")
        exit()

    # Weight Initialization ## what is weight_init?
    model.apply(weight_init)

    # Making Initial Mask
    make_mask(model)

    ## Scale the network weights unproportionally
    original_initialization(mask,copy.deepcopy(model.state_dict()))
    ## model changes here
    initial_state_dict = copy.deepcopy(model.state_dict())
    ## print model paramaters by layers


    # Copying and Saving Initial State
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/")
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/initial_state_dict_{args.prune_type}.pth.tar")


    # Optimizer and Loss
    if args.er == "SAM":
        base_opt = torch.optim.SGD        
        optimizer = SAM(model.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay=1e-4)
    else:    
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size(),torch.mean(torch.abs(param.data)))

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)
    noise_type=args.noise_type
    prior_sigma = args.prior
    noise_step = args.noise_step
    kl=args.kl



    for _ite in range(args.start_iter, ITERATION):
        
        if not _ite == 0:
            if args.prune_type=="noise":
                prune_by_noise(args.prune_percent, train_loader,criterion,noise_type,prior_sigma,kl,num_steps=noise_step)
            else:    
                prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                step = 0
                for name, param in model.named_parameters():
                    #if 'weight' in name:
                    weight_dev = param.device
                    param.data =  param.data * mask[step].to(weight_dev)
                    step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)

        count_nonzero(model,mask)        

        if args.er == "SAM":
            base_opt = torch.optim.SGD        
            optimizer = SAM(model.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay=1e-4)
        else:    
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)           

        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        #utils.plot_and_save_mask(mask, _ite,args)

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))


        ## Training starts with the pruned weights here 

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            loss = train(model, train_loader, optimizer, criterion, args.er, args.reg)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            #torch.save(model.state_dict(), 'model.pth')
            #model.load_state_dict(torch.load('model.pth'))
            
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        # plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
        # plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
        # plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})") 
        # plt.xlabel("Iterations") 
        # plt.ylabel("Loss and Accuracy") 
        # plt.legend() 
        # plt.grid(color="gray") 
        # utils.checkdir(f"{os.getcwd()}/plots/{args.prune_type}/{args.arch_type}/{args.dataset}/acc")
        # plt.savefig(f"{os.getcwd()}/plots/{args.prune_type}/{args.arch_type}/{args.dataset}/acc/{args.prune_type}_LossVsAccuracy_{comp1}.png") 
        # plt.close()

        # # Dump Plot values
        # utils.checkdir(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset}/")
        # all_loss.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        # all_accuracy.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")
        
        # # Dumping mask
        # utils.checkdir(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset}/")
        # with open(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
        #     pickle.dump(mask, fp)
        
        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/")
    comp.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type},{args.prune_type}, Kl:{args.kl}, Prior:{args.prior})")
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/plots/{args.prune_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/acc")
    plt.savefig(f"{os.getcwd()}/plots/{args.prune_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/acc/{args.prune_type}KL{args.kl}P{args.prior}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()                    
   
# Function for Training
def train(model, train_loader, optimizer, criterion, ir,reg):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
            #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()                
                    
        step=0
        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            #if 'weight' in name:
            grad_tensor = p.grad.data*mask[step]
            p.grad.data = grad_tensor.to(device)
            step+=1
        optimizer.step()
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False,**kwargs):
    global step
    global mask
    global model

    # Calculate percentile value
    all_alive_weights = []
    for name, param in model.named_parameters():
        # We prune bias term
        tensor = param.data.cpu().numpy()
        alive = np.abs(tensor[np.nonzero(tensor)])  # flattened array of nonzero values
        all_alive_weights.extend(alive)

    # Get the percentile value from all weights (as opposed to only layerwise)
    percentile_value = np.percentile(all_alive_weights, percent)
    percentile_value_torch = torch.tensor(percentile_value).to(next(model.parameters()).device)
    ##print the percentile value
    print(f'Pruning with threshold : {percentile_value}')
    # Now prune the weights
    step = 0
    for name, param in model.named_parameters():
        tensor = param.data
        weight_dev = param.device
        #new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
        new_mask = torch.where(tensor.abs() < percentile_value_torch, torch.tensor(0., device=weight_dev), mask[step])                
        # Apply new weight and mask
        param.data = (tensor * new_mask).to(weight_dev)
        mask[step] = new_mask
        step += 1
    step = 0




def prune_by_noise(percent,train_loader,criterion, noise_type ,prior_sigma=1.0, kl="no", num_steps=25):
    global model
    global mask
    global step 
    EPS=1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kl_loss = torch.zeros(1, device=device)
    torch.manual_seed(0)

    numel = sum(param.numel() for param in model.parameters())

    _,p,_ ,prior= initialization(model,prior_sigma,noise_type)

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
                #print(param)


            ## no noise added at the pruned locations
            if noise_type=="gaussian":
                k=0
                step=0
                for i, param in enumerate(model_copy.parameters()):
                    with torch.no_grad():
                        mask_torch = mask[step].to(device)

                    t = len(param.view(-1))
                    eps = torch.randn_like(param.data).to(device)                
                    noise = torch.reshape(torch.exp(p[k:(k+t)]),param.data.size()) * eps  * mask_torch
                    with torch.no_grad(): 
                        if batch_idx==0:
                            print(torch.mean(torch.abs(param.data).view(-1)/torch.exp(p[k:(k+t)])))                                         
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
                        mask_torch = mask[step].to(device)
                    t = len(param.view(-1))
                    logits = torch.reshape(p[k:(k+t)], param.data.size()).to(device)
                    #print(mask_torch.shape,logits.shape,torch.sigmoid(logits).shape)
                    noise = generate_noise_soft(torch.sigmoid(logits),temp=0.2) *mask_torch
                    # with torch.no_grad():
                    #      if batch_idx == 0:
                             ##plot histogram of noise values with figure name having iteration number
                            # plt.hist(noise.cpu().numpy().flatten(), bins=100)                             
                            # plt.savefig(f"{os.getcwd()}/plots/{args.prune_type}/{args.arch_type}/{args.dataset}/noise_hist_{epoch}_{i}.png")
                            # plt.title(f"noise_hist_{epoch}_{i}")
                            # plt.close()

                            # plt.hist(logits.cpu().numpy().flatten(), bins=100)
                            # plt.savefig(f"{os.getcwd()}/plots/{args.prune_type}/{args.arch_type}/{args.dataset}/logits_hist_{epoch}_{i}.png")
                            # plt.title(f"logits_hist_{epoch}_{i}")
                            # plt.close()
                            #print(f"This is p[{k}:{k+t}]:", p[k:k+t])
                            #print("This is noise:", noise)
                            ## priint mean of  p[{k}:{k+t}]:", p[k:k+t]
                            # print("This is mean of noise:", torch.mean(noise))
                            # print(f"This is mean of p[{k}:{k+t}]:", torch.mean(p[k:k+t]))
                        
            
                    k += t
                    step +=1
                    param.mul_(noise)
                if kl=="yes":
                    kl_loss = (torch.sigmoid(p) * torch.log((torch.sigmoid(p)+1e-6)/prior) + (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p)+1e-6)/(1-prior))).sum()
     
            # # Forward pass after adding noise
            output = model_copy(data)
            batch_original_loss_after_noise = criterion(output, target)


            if kl=="yes":
                total_loss =  batch_original_loss_after_noise + 1e-3* kl_loss
            else:                    
                total_loss =  batch_original_loss_after_noise

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
                if kl=="yes":
                    kl_loss_accum += kl_loss.item()
                batch_original_loss_after_noise_accum += batch_original_loss_after_noise.item()


        # Average losses for the mini-batch
        print(f"Epoch {epoch+1}")
        print(f"Average batch original loss after noise: {batch_original_loss_after_noise_accum / len(train_loader):.6f}")
        if kl=="yes":
            print(f"Average KL loss: {kl_loss_accum / len(train_loader):.6f}")
        print(f"Average total loss: {total_loss_accum / len(train_loader):.6f}")

        #print(f'p optimization step, batch original loss after noise: {batch_original_loss_after_noise_avg:.6f}, Total Loss: {total_loss_avg:.6f}, KL Loss: {kl_loss_avg:.6f}')
    if noise_type=="gaussian":
        with torch.no_grad():    
            k=0
            step=0
            # Flatten all weights into a single list
            all_normalized_tensors = []
            for i, (name, param) in enumerate(model.named_parameters()):
                t = len(param.view(-1))
                tensor = param.data.cpu().numpy()
                normalized_tensor = np.abs(tensor) /  torch.reshape(torch.exp(p[k:(k+t)]), tensor.shape).cpu().detach().numpy()
                alive = normalized_tensor[np.nonzero(normalized_tensor)]
                all_normalized_tensors.extend(alive)
                k += t
            # Get the percentile value from all weights (as opposed to only layerwise)
            percentile_value = torch.tensor(np.percentile(all_normalized_tensors, percent)).to(next(model.parameters()).device)

            # Now prune the weights
            k = 0
            step = 0
            for i, (name, param) in enumerate(model.named_parameters()):
                t = len(param.view(-1))
                tensor = param.data.cpu().numpy()
                #normalized_tensor = np.abs(tensor) /  torch.reshape(torch.exp(p[k:(k+t)]), tensor.shape).cpu().detach().numpy()
                normalized_tensor = torch.abs(param.data) / torch.exp(p[k:(k + t)]).view(param.shape)
                weight_dev = param.device
                #new_mask = np.where(normalized_tensor < percentile_value, 0, mask[step])
                new_mask = torch.where(normalized_tensor < percentile_value, torch.zeros_like(param.data).to(weight_dev), mask[step])
                        
                # Apply new weight and mask
                #param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                param.data *= new_mask
                mask[step] = new_mask
                k += t
                step += 1
            step = 0
            k=0

    elif noise_type == "bernoulli":
        with torch.no_grad():
            p_values = p.data
            unpruned_p_values = []
            k = 0
            for m in mask:
                t = m.numel() # total elements in current layer
                layer_p_values = p_values[k:(k + t)]  # get p_values for the current layer
                unpruned_indices = torch.nonzero(m.flatten()).squeeze()  # unpruned indices for the current layer
                unpruned_p_values.extend(layer_p_values[unpruned_indices].tolist())
                k += t

            # Get the percentile value from all p values
            percentile_value = torch.tensor(np.percentile(unpruned_p_values, percent)).to(next(model.parameters()).device)
            print(f"Pruning iteration {epoch + 1}, Percentile value: {percentile_value}")

            # Pruning the weights
            k = 0
            step = 0
            for i, (name, param) in enumerate(model.named_parameters()):
                t = len(param.view(-1))
                tensor = param.data
                weight_dev = param.device
                new_mask = torch.where(torch.reshape(p_values[k:(k + t)], param.shape) < percentile_value, torch.tensor(0).to(weight_dev), mask[step].clone().detach().to(weight_dev))  # Prune based on reshaped p_values

                # Apply new weight and mask
                param.data = tensor * new_mask
                mask[step] = new_mask
                k += t
                step += 1




def initialization(model,prior_sigma,noise_type ="gaussian", w0decay=1.0):
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
        #prior_sigma = torch.log(w0.abs().mean())
        #prior = torch.log(w0.abs())
        prior = torch.where(w0 == 0, torch.zeros_like(w0), torch.log(torch.abs(w0)))
    elif noise_type=="bernoulli":
        print("here")
        p = nn.Parameter(torch.zeros_like(w0), requires_grad=True)
        prior = torch.sigmoid(prior_sigma*torch.ones_like(w0))
        
    return w0, p, num_layer,prior

# Function to make an empty mask of the same size as the model weights 
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        #if 'weight' in name:
        step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        #if 'weight' in name:
        tensor = param.data
        mask[step] = torch.ones_like(tensor)
        step = step + 1
    step = 0

def original_initialization(mask_temp, initial_state_dict):
    global model    
    step = 0
    for name, param in model.named_parameters(): 
        #if "weight" in name: 
        weight_dev = param.device
        param.data = (mask_temp[step] * initial_state_dict[name]).to(weight_dev)
        #param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
        step = step + 1
        #if "bias" in name:
        #    param.data = initial_state_dict[name]
    step = 0

def scale_initialization(mask_temp, initial_state_dict):
    global model    
    step = 0
    for name, param in model.named_parameters(): 
        #if "weight" in name: 
        weight_dev = param.device
        param.data = (step+1)*(mask_temp[step] * initial_state_dict[name]).to(weight_dev)
        #param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
        step = step + 1
        #if "bias" in name:
        #    param.data = initial_state_dict[name]
    step = 0

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def generate_noise_soft(logits,temp=0.5):
    gumbel1 = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)
    gumbel2 = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)    
    numerator = torch.exp((logits + gumbel1)/temp)
    denominator = torch.exp((logits + gumbel1)/temp)  + torch.exp(((1 - logits) + gumbel2)/temp)    
    noise = numerator / denominator
    return noise


def count_nonzero(model, mask):
    nonzero_model = 0
    nonzero_mask = 0
    total = 0
    mask_count = 0

    for i, (name, param) in enumerate(model.named_parameters()):
        tensor = param.data

        # Calculate non-zeros in the model parameters and the mask
        nonzero_model += torch.count_nonzero(tensor).item()
        nonzero_mask += torch.count_nonzero(mask[i]).item()

        # Calculate total elements in the model parameters and the mask
        total += torch.numel(tensor)
        mask_count += torch.numel(mask[i])

    # Print a statement that prints the percentage of nonzero weights in the model and the mask
    print(f"Non-zero model percentage: {nonzero_model / total * 100}%, Non-zero mask percentage: {nonzero_mask / mask_count * 100}%")





if __name__=="__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 0.1, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=10, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit|noise")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121|fcs")
    parser.add_argument("--prune_percent", default=80, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=10, type=int, help="Pruning iterations count")
    parser.add_argument("--er", default=None, type=str, help="type of regularization")
    parser.add_argument("--reg", default=0.1, type=float , help="regularization strength")
    parser.add_argument("--noise_type", default="gaussian", type=str , help="chose gaussian or bernoulli noise")
    parser.add_argument("--prune_noise_iter", default=50, type=int , help="number of iterations for pruning by noise")
    parser.add_argument("--kl", default="yes", type=str , help="if kl should be used or not")
    parser.add_argument("--prior", default=0.0, type=float , help="prior centre in kl")
    parser.add_argument("--noise_step", default=10, type=int , help="number of noise iterations")
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)