#ANCHOR Libraries
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy

#ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero/total)*100,1))

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


def plot_and_save_mask(mask, prune_iter,args):
    for i, layer_mask in enumerate(mask):
        plt.figure(figsize=(10, 10))

        # Check if the layer mask is 1D (e.g., for bias)
        if len(layer_mask.shape) == 1:
            # For 1D arrays, use a bar plot
            plt.bar(np.arange(len(layer_mask)), layer_mask)
        else:
            # For 2D arrays, use imshow
            plt.figure(figsize=(10,3))  # Increase figure size
            plt.imshow(layer_mask, cmap='gray', aspect='auto')  # Change aspect ratio

        checkdir(f"{os.getcwd()}/plots/{args.prune_type}/{args.arch_type}/{args.dataset}/mask")
        plt.title(f"Pruning iteration {prune_iter}, Layer {i+1}")
        #plt.colorbar()
        #plt.savefig(f"{prune_type}_mask_plot_iter_{prune_iter}_layer_{i+1}.png")
        plt.savefig(f"{os.getcwd()}/plots/{args.prune_type}/{args.arch_type}/{args.dataset}/mask/{args.prune_type}_mask_plot_iter_{prune_iter}_layer_{i+1}.png", dpi=1200) 
        plt.show()
        plt.close()



def torch_percentile(t, percent):
    if not isinstance(t, torch.Tensor):
        if isinstance(t, list):
            t = torch.cat(t)
        else:
            raise TypeError("Input must be a Tensor or a list of Tensors")
    
    if not (0 <= percent <= 100):
        raise ValueError("percent must be in the range [0, 100]")

    sorted_t = torch.sort(t).values
    index = int(percent/100. * (t.numel()-1))
    return sorted_t[index]
      


#ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)

#FIXME 
def plot_train_test_stats(stats,
                          epoch_num,
                          key1='train',
                          key2='test',
                          key1_label=None,
                          key2_label=None,
                          xlabel=None,
                          ylabel=None,
                          title=None,
                          yscale=None,
                          ylim_bottom=None,
                          ylim_top=None,
                          savefig=None,
                          sns_style='darkgrid'
                          ):

    assert len(stats[key1]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key1, len(stats[key1]), epoch_num)
    assert len(stats[key2]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key2, len(stats[key2]), epoch_num)

    plt.clf()
    sns.set_style(sns_style)
    x_ticks = np.arange(epoch_num)

    plt.plot(x_ticks, stats[key1], label=key1_label)
    plt.plot(x_ticks, stats[key2], label=key2_label)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if yscale is not None:
        plt.yscale(yscale)

    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        plt.ylim(top=ylim_top)

    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fancybox=True)

    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()

        