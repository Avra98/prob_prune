# Importing Libraries
import argparse
import os
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
#from tensorboardX import SummaryWriter

# Custom Libraries
from utility.data import *
from utility.log_utils import *
from utility.func_utils import *
from utility.prune import *
from utility.prune_pac import *

# Tensorboard initialization
#writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    model_init = copy.deepcopy(model)
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
    end_iter = args.end_iter
    all_loss = np.zeros(end_iter,float)
    all_accuracy = np.zeros(end_iter,float)

    noise_type=args.noise_type
    prior_sigma = args.prior
    kl=args.kl
    noise_step = args.noise_step
    
    p_old = None
    for _ite in range(args.start_iter, ITERATION):
        
        if not _ite == 0:
            ## prune here 
            if args.prune_type=="noise":

                mask, p_new = prune_by_noise(model, mask, args.prune_percent, dataset.train,criterion,noise_type,
                	   prior_sigma,kl,num_steps=noise_step,lr=args.lr_p, p_init=p_old)
                if args.initial_p == "last":
                    p_old = p_new.detach().clone()

            elif args.prune_type=="noise_pac":
                # reweight weight
                with torch.no_grad():
                    for i, (param1, param2) in enumerate(zip(model.parameters(), model_init.parameters())):
                        param2.data = (torch.norm(param1.data * mask[i]) / 
                                                    (torch.norm(param2.data * mask[i])+1e-6) * 
                                                    param2.data * mask[i])

                mask, p_new  = prune_by_noise_trainable_prior(model, model_init, mask, args.prune_percent, dataset.train,criterion,
                    num_steps=noise_step,lr=args.lr_p, p_init=p_old)   
                if args.initial_p == "last":
                    p_old = p_new.detach().clone()
                                 
            elif args.prune_type=="lt":    
                mask = prune_by_percentile(model, mask, args.prune_percent)
            elif args.prune_type=="random":
                mask = prune_by_random(model, mask, args.prune_percent)


            ## initialize here 
            if args.initial=="reinit":
                reset_all_weights(model)
                step = 0
                for param in model.parameters():
                    if param.requires_grad:
                        param.data = param.data * mask[step]
                        step += 1
            elif args.initial=="original":
                original_initialization(model, mask, model_init)
            elif args.initial=="last":
                original_initialization(model, mask, copy.deepcopy(model)) ## does not alter the model, only masks it 
            elif args.initial=="rewind":
                print("initialization at rewind")
                ## load the model from the rewind folder
                model_rewind = torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/rewind/{args.kl}+{args.prior}/{args.rewind_iter}_model_{args.prune_type}_{args.noise_type}.pth.tar")
                ## mask the model
                original_initialization(model, mask, copy.deepcopy(model_rewind))

        count_nonzero(model, mask)   
        if args.optimizer.lower() == "sgd":         
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                            momentum = args.momentum,
            								 weight_decay = args.weight_decay)
        else:     
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                         weight_decay = args.weight_decay)           

        print(f"\n--- Pruning Level [{_ite}/{ITERATION}]: ---")
        # Print the table of Nonzeros in each layer
        comp1 = print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(end_iter))

        torlence_iter = 0
        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, dataset.test, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torlence_iter = 0
                    checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/{_ite}_model_{args.prune_type}.pth.tar")
                else:
                    torlence_iter += 1

                if torlence_iter > 10:
                    break

            # Training
            if args.noise_std == 0:
                loss = train(model, mask, dataset.train, optimizer, criterion)
            else:
                loss = train_with_noise(model, mask, dataset.train, optimizer, criterion, 
                                            noise_std=args.noise_std, noise_type=args.inject_noise)

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
                    f'Train Epoch: {iter_}/{end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

        print(f'Train Epoch: {iter_}/{end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%') 
        #writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(end_iter,float)
        all_accuracy = np.zeros(end_iter,float)
        
        checkdir(f"{os.getcwd()}/dumps/{args.prune_type}/{args.noise_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/")
        all_loss.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.noise_type}/{args.arch_type}/{args.dataset}/{args.kl}+{args.prior}/{args.prune_type}_all_loss{str(_ite)}.dat")

    # Dumping Values for Plotting
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
    parser.add_argument("--optimizer", "-opt", default="SGD", help="optimizer")
    parser.add_argument("--momentum", default=0.0, type=float, help="momentum for SGD")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight Decay")
    parser.add_argument("--lr_p", default=1e-2, type=float, help="lr for posterier variance")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--start_iter", default=0, type=int) 
    parser.add_argument("--end_iter", default=25, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", default=1, type=int, help="number of threads of data loader")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt |noise|random")
    
    parser.add_argument("--initial", default="last", type=str, help="reinit|original|last|rewind")
    parser.add_argument("--initial_p", default="last", type=str, help="reinit|last")

    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121|fcs")
    parser.add_argument("--prune_percent", default=0.8, type=float, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=6, type=int, help="Pruning iterations count")
    parser.add_argument("--noise_type", default="gaussian", type=str , help="chose gaussian or bernoulli noise")
    parser.add_argument("--kl", default=1e-4, type=float, help="if using the kl term")
    parser.add_argument("--augmentation", "-aug", action='store_true', help="if using augmentation.")
    
    parser.add_argument("--noise_std", default=0, type=float, help="if using nosie injection during training.")
    parser.add_argument("--inject_noise", default="iso", type=str, help="noise type for noise injection: isotropic (iso) | anisotropic (ani) ")

    parser.add_argument("--prior", default=0.0, type=float , help="prior centre in kl")
    parser.add_argument("--noise_step", default=10, type=int , help="number of noise iterations")
    parser.add_argument("--rewind_iter", default=3, type=int , help="number of rewind iterations")
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    main(args)
