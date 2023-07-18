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

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet 

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
    
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
    # If you want to add extra model paste here
    else:
        print("\nWrong Model choice\n")
        exit()

    # Weight Initialization ## what is weight_init?

    model.apply(weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss
    if args.er == "SAM":
        base_opt = torch.optim.SGD        
        optimizer = SAM(model.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay=1e-4)
    else:    
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

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



    for _ite in range(args.start_iter, ITERATION):
        if not _ite == 0:
            if args.prune_type=="noise":
                prune_by_noise(args.prune_percent,train_loader,criterion)
            else:    
                prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)

        if args.er == "SAM":
            base_opt = torch.optim.SGD        
            optimizer = SAM(model.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay=1e-4)
        else:    
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)           

        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

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
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            loss = train(model, train_loader, optimizer, criterion, args.er, args.reg)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy
            
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
        plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})") 
        plt.xlabel("Iterations") 
        plt.ylabel("Loss and Accuracy") 
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200) 
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")
        
        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)
        
        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()                    
   
# Function for Training
def train(model, train_loader, optimizer, criterion, ir,reg):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):

        if ir=="SAM":
            #imgs, targets = next(train_loader)
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            train_loss = criterion(output, targets)
            train_loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(imgs), targets).backward()
            optimizer.second_step(zero_grad=True)
        elif ir=="Jac":    
            optimizer.zero_grad()
            #imgs, targets = next(train_loader)
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            train_loss = criterion(output, targets)
            train_loss.backward(retain_graph=True, create_graph=True)
            dot=0.0
            norm_square_sum=0.0
            v = torch.randn(output.shape, requires_grad=False).cuda()
            dot = output.mul(v).sum()/(args.batch_size)
            grads = torch.autograd.grad(dot, inputs=model.parameters(), create_graph=True)                
            for g in grads:
                norm_square_sum += torch.norm(g) ** 2   
            implicit = reg*norm_square_sum
            implicit.backward()
        else:
            optimizer.zero_grad()
            #imgs, targets = next(train_loader)
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            train_loss = criterion(output, targets)
            train_loss.backward()                
            


        

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
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
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
                
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        step = 0



def prune_by_noise(percent,train_loader,criterion, prior_sigma=1.0,lr=1e-3, num_steps=1000):
    global model
    global mask
    global step 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    num_parameters = 0
    for param in model.parameters():
        prior_sigma += torch.abs(param).sum().item()
        num_parameters += param.numel()
    prior_sigma /= num_parameters
    print(prior_sigma)
    # Learn noise level
    # Initialize p for each model parameter
  # Learn noise level
    #p = [torch.ones_like(param).to(device) * 0.5 * np.log(prior_sigma) for param in model.parameters()]
    #p = [nn.Parameter(torch.ones_like(param).to(device) * 0.5 * np.log(prior_sigma)) for param in model.parameters()]
    _,p,_ = initialization(model)


    #for tensor in p:
    #    tensor.requires_grad_(True)

    optimizer_p = torch.optim.Adam([p], lr=lr)

    for _ in range(num_steps):
        # Initialize accumulators
        batch_original_loss_after_noise_accum = 0.0
        total_loss_accum = 0.0
        kl_loss_accum = 0.0

        # Loop over mini-batches
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass before adding noise
            output = model(data)
            batch_original_loss = criterion(output, target)
            #print(f"Original Loss before adding noise: {batch_original_loss.item()}")

            optimizer_p.zero_grad()
            model_copy = copy.deepcopy(model)

            #for i, (name, param) in enumerate(model_copy.named_parameters()):
            #    if 'weight' in name:
            #        eps = torch.randn_like(param).to(device)
            #        noise = torch.exp(p[i]) * eps  
            #        #param.data.add_(noise) 
            #        param=param+noise
            k=0
            for i, param in enumerate(model.parameters()):
                t = len(param.view(-1))
                eps = torch.randn_like(param.data).to(device)                
                noise = torch.reshape(torch.exp(p[k:(k+t)]),param.data.size())* eps  
                 
                param = param+noise
                print(noise)
                k += t        

                    #with torch.no_grad(): 
                    #    if batch_idx==0:
                    #        print(torch.norm(param.data),torch.norm(noise))
            # Forward pass after adding noise
            output = model(data)
            batch_original_loss_after_noise = criterion(output, target)
            batch_original_loss_after_noise_accum += batch_original_loss_after_noise.item()

            
            #print(f"Original Loss after adding noise: {batch_original_loss_after_noise.item()}")

            total_loss = batch_original_loss_after_noise
            kl_loss = torch.zeros(1, device=device)

           # for i, (name, param) in enumerate(model.named_parameters()):
           #     if 'weight' in name:
           #         kl_div = 0.5 * (torch.sum(torch.exp(2*p[i])/prior_sigma - 2*p[i]) - param.numel() - param.numel() * np.log(prior_sigma))
           #         kl_loss += kl_div.

            #total_loss +=0*kl_loss
            #with torch.no_grad():
            #    total_loss_accum += total_loss.item()
            #    kl_loss_accum += kl_loss
            #print(kl_loss)

            total_loss.backward()
            #kl_loss.backward()
            print(p[0].grad)
            optimizer_p.step()

        # Average losses for the mini-batch
        batch_original_loss_after_noise_avg = batch_original_loss_after_noise_accum / len(train_loader)
        total_loss_avg = total_loss_accum / len(train_loader)
        kl_loss_avg = kl_loss_accum / len(train_loader)

        print(f'p optimization step, batch original loss after noise: {batch_original_loss_after_noise_avg:.6f}, Total Loss: {total_loss_avg:.6f}, KL Loss: {kl_loss_avg:.6f}')







    step=0
    # Prune weights based on learned noise level
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            # Normalize weights with standard deviation of noise
            normalized_tensor = np.abs(tensor) / np.exp(p[step].cpu().detach().numpy())
            alive = normalized_tensor[np.nonzero(normalized_tensor)] # flattened array of nonzero values
            percentile_value = np.percentile(alive, percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(normalized_tensor < percentile_value, 0, mask[step])
                
            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0

def initialization(model, w0decay=1.0):
    for param in model.parameters():
        param.data *= w0decay

    device = next(model.parameters()).device
    w0 = []
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone())
    num_layer = layer + 1
    w0 = torch.cat(w0) 
    p  = nn.Parameter(torch.ones(len(w0), device=device)*torch.log(w0.abs().mean()), requires_grad=True)
    #we = nn.Parameter(torch.ones(1, device=device)*torch.log(w0.abs().mean()), requires_grad=True)
    return w0, p, num_layer

# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
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


if __name__=="__main__":
    
    #from gooey import Gooey
    #@Gooey      
    
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 0.1, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=1280, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=10, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit|noise")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=8, type=int, help="Pruning iterations count")
    parser.add_argument("--er", default=None, type=str, help="type of regularization")
    parser.add_argument("--reg", default=0.1, type=float , help="regularization strength")
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
