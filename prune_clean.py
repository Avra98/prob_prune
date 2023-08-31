# Importing Libraries
import argparse
import os
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn

from tensorboardX import SummaryWriter

# Custom Libraries
from utility.data import *
from utility.log_utils import *
from utility.func_utils import *
from utility.prune import *
from utility.prune_pac import *

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

torch.manual_seed(0)
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.initial=="reinit" else False

    # Data Loader 
    if args.dataset.lower() == "cifar10":
        size = (32, 32)
        dataset = Cifar10(args.batch_size, args.threads, size, args.augmentation)
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet ,fcs
    elif args.dataset.lower() == "fashionmnist":
        size = (28, 28)
        dataset = FashionMNIST(args.batch_size, args.threads, size, args.augmentation)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet
    elif args.dataset.lower() == "cifar100":
        size = (32, 32)
        dataset = Cifar100(args.batch_size, args.threads, size, args.augmentation)
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  ,fcs
    else:
        size = (28, 28)
        dataset = MNIST(args.batch_size, args.threads, size, args.augmentation)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet, fcs

    # Model Loader
    if args.arch_type.lower() == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type.lower() == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type.lower() == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type.lower() == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type.lower() == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type.lower() == "densenet121":
        model = densenet.densenet121().to(device)   
    elif args.arch_type.lower() == "fcs":
        model = fcs.fcs().to(device)    
    else:
        print("\nWrong Model choice\n")
        exit()

    # Making Initial Mask
    mask = make_mask(model)

    # Initial Model
    initial_state_dict = copy.deepcopy(model.state_dict())

    # Copying and Saving Initial State
    checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/")
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/initial_state_dict_{args.prune_type}.pth.tar")
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

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
    kl=args.kl
    noise_step = args.noise_step
    
    for _ite in range(args.start_iter, ITERATION):
        
        if not _ite == 0:
            ## prune here 
            if args.prune_type=="noise":
                prune_by_noise(model, mask, args.prune_percent, dataset.train,criterion,noise_type,
                	prior_sigma,kl,num_steps=noise_step,lr=args.lr_p)
            elif args.prune_type=="noise_pac":
                prune_by_noise_trainable_prior(model, mask, args.prune_percent, dataset.train,criterion,noise_type,
                    num_steps=noise_step,lr=args.lr_p)                
            elif args.prune_type=="lt":    
                prune_by_percentile(model, mask, args.prune_percent)
            elif args.prune_type=="random":
                prune_by_random(model, mask, args.prune_percent)


            ## initialize here 
            if args.initial=="reinit":
                reset_all_weights(model)
                step = 0
                for param in model.parameters():
                    if param.requires_grad:
                        param.data = param.data * mask[step]
                        step += 1
            elif args.initial=="original":
                original_initialization(model, mask, initial_state_dict)
            elif args.initial=="last":
                original_initialization(model, mask, copy.deepcopy(model.state_dict())) ## does not alter the model, only masks it 
            elif args.initial=="rewind":
                print("initialization at rewind")
                ## load the model from the rewind folder
                model_rewind = torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/rewind/{args.kl}+{args.prior}/{args.rewind_iter}_model_{args.prune_type}_{args.noise_type}.pth.tar")
                ## mask the model
                original_initialization(model, mask, copy.deepcopy(model_rewind.state_dict()))

        count_nonzero(model, mask)           
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
        								momentum=0.9, weight_decay=5e-4)           

        print(f"\n--- Pruning Level [{_ite}/{ITERATION}]: ---")
        # Print the table of Nonzeros in each layer
        comp1 = print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, dataset.test, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            loss = train(model, mask, dataset.train, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            if args.initial=="rewind" and args.rewind_iter==iter_ and _ite==0:
                ## save the model in a separaate folder named rewind
                checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/rewind/{args.kl}+{args.prior}")
                torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/rewind/{args.kl}+{args.prior}/{args.rewind_iter}_model_{args.prune_type}_{args.noise_type}.pth.tar")
                print("here at",iter_)
            
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)

    # Dumping Values for Plotting
    checkdir(f"{os.getcwd()}/dumps/{args.prune_type}/{args.noise_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/")
    comp.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.noise_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.noise_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/{args.prune_type}_bestaccuracy.dat")

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
    checkdir(f"{os.getcwd()}/plots/{args.prune_type}/{args.initial}/{args.noise_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/acc")
    plt.savefig(f"{os.getcwd()}/plots/{args.prune_type}/{args.initial}/{args.noise_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/acc/{args.prune_type}KL{args.kl}P{args.prior}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()                    




if __name__=="__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 0.02, type=float, help="Learning rate")
    #parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight Decay")
    parser.add_argument("--lr_p", default=1e-2, type=float, help="lr for posterier variance")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--start_iter", default=0, type=int) 
    parser.add_argument("--end_iter", default=10, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", default=4, type=int, help="number of threads of data loader")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt |noise|random")
    parser.add_argument("--initial", default="reinit", type=str, help="reinit|original|last|rewind")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121|fcs")
    parser.add_argument("--prune_percent", default=0.80, type=float, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=4, type=int, help="Pruning iterations count")
    parser.add_argument("--noise_type", default="gaussian", type=str , help="chose gaussian or bernoulli noise")
    parser.add_argument("--kl", action='store_true', help="if using the kl term")
    parser.add_argument("--augmentation", "-aug", action='store_true', help="if using augmentation.")
    parser.add_argument("--prior", default=0.0, type=float , help="prior centre in kl")
    parser.add_argument("--noise_step", default=10, type=int , help="number of noise iterations")
    parser.add_argument("--rewind_iter", default=3, type=int , help="number of rewind iterations")
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    main(args)
