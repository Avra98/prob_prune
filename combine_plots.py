import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

DPI = 1200
prune_iterations = 5
arch_types = ["fc1"]
prune_type=["lt","noise","random"]
noise_type=["bernoulli","gaussian"]
datasets = ["mnist", "fashionmnist"]

sns.set_style('darkgrid')

for arch_type in tqdm(arch_types):
    
    plt.figure(figsize=(12, 6))  # Set the figure size
    
    for idx, dataset in enumerate(datasets, start=1):  # enumerate to get index for subplot
        d_lt = np.load(f"{os.getcwd()}/dumps/lt/gaussian/{arch_type}/{dataset}/0.0001+0.0/lt_compression.dat", allow_pickle=True)
        b_lt = np.load(f"{os.getcwd()}/dumps/lt/gaussian/{arch_type}/{dataset}/0.0001+0.0/lt_bestaccuracy.dat", allow_pickle=True)

        d_random = np.load(f"{os.getcwd()}/dumps/random/gaussian/{arch_type}/{dataset}/0.0001+0.0/random_compression.dat", allow_pickle=True)
        b_random = np.load(f"{os.getcwd()}/dumps/random/gaussian/{arch_type}/{dataset}/0.0001+0.0/random_bestaccuracy.dat", allow_pickle=True)

        a = np.arange(prune_iterations)
        
        plt.subplot(1, 2, idx)  # 1 row, 2 columns, index=idx

        plt.plot(a, b_lt, '-o', c="blue", label="Magnitude pruning", linewidth=2, markersize=8, markerfacecolor="white", markeredgecolor="blue")
        plt.plot(a, b_random, '-*', c="red", label="Random pruning", linewidth=2, markersize=8, markerfacecolor="white", markeredgecolor="red")
        
        plt.title(f"{dataset}", fontsize=15)
        plt.xlabel("Unpruned Weights %", fontsize=14, weight='bold')
        plt.ylabel("Test accuracy %", fontsize=14, weight='bold')
        plt.xticks(a, d_lt, rotation="vertical", fontsize=12, weight='bold')
        plt.yticks(fontsize=12, weight='bold')
        plt.ylim(0,100)
        plt.legend(fontsize=12, loc='lower left')
        plt.grid(color="gray", linestyle="--", linewidth=0.5)  # Here's the dashed grid
    
    plt.tight_layout()  # Adjusts subplot layout
    plt.savefig(f"{os.getcwd()}/plots/combined_{arch_type}.png", dpi=DPI, bbox_inches='tight') 
    plt.close()

