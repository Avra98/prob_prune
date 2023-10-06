#ANCHOR Libraries
import numpy as np
import torch
import os
#import seaborn as sns
import matplotlib.pyplot as plt
import copy

#ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        nz_count = torch.count_nonzero(param.data)
        total_params = param.data.numel() 
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} \
            ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {param.data.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero.item()/total)*100,1))

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

        
