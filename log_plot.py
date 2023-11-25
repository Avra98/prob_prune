



import matplotlib.pyplot as plt
import re
import os

# Directory where your log files are located
log_files_directory = 'TEST_fc1_testprune_new_gumbel_sgd+adam_second_0.2'

# Regular expression patterns to extract hyperparameters for both bernoulli and lt files
bernoulli_pattern = r'bernoulli_kl(\S+)_data(\S+)_arch(\S+)_percent(\S+)_lrp(\S+)_nt(sparse_)?bernoulli_opt(\S+).out'
lt_pattern = r'lt_data(\S+)_arch(\S+)_percent(\S+)_opt(\S+).out'
accuracy_pattern = r'Best test Accuracy: (\d+\.\d+)%'

# Initialize dictionaries to hold the data and titles
data_dict = {}
titles_dict = {}

# Read the Bernoulli files
for filename in os.listdir(log_files_directory):
    if filename.endswith('.out') and 'bernoulli' in filename:
        # Extract hyperparameters from the filename
        hyperparams_match = re.search(bernoulli_pattern, filename)
        if hyperparams_match:
            kl, data, arch, percent, lrp, nt_type, opt = hyperparams_match.groups()
            nt = 'nt' + (nt_type if nt_type else '') + 'bernoulli'
            if data.lower() != 'fashionmnist':
                continue
            if opt.lower() != 'adam':
                continue
            hyperparam_str = f'kl{kl}_lrp{lrp}_{nt}'
            title_str = f'Data: {data}, Arch: {arch}, Pruning Percent: {percent}, Optimizer: {opt}'

            titles_dict[hyperparam_str] = title_str
            with open(os.path.join(log_files_directory, filename), 'r') as file:
                log_contents = file.read()
                best_test_accuracy_matches = re.findall(accuracy_pattern, log_contents)
                best_test_accuracies = [float(match) for match in best_test_accuracy_matches]
                data_dict[hyperparam_str] = best_test_accuracies

# Read the LT file and add it to the dictionaries
lt_filename = 'lt_datafashionmnist_archfc1_percent0.2_optAdam.out'
with open(os.path.join(log_files_directory, lt_filename), 'r') as file:
    log_contents = file.read()
    best_test_accuracy_matches = re.findall(accuracy_pattern, log_contents)
    best_test_accuracies = [float(match) for match in best_test_accuracy_matches]
    # Using 'lt' as the hyperparameter string for LT pruning
    data_dict['lt'] = best_test_accuracies

# Plotting all in one figure
plt.figure(figsize=(10, 8))
# Assuming all files have the same data, arch, percent, and optimizer
title_keys = list(titles_dict.keys())
if title_keys:
    plt.title(titles_dict[title_keys[0]])
for hyperparam_str, accuracies in data_dict.items():
    pruning_levels = list(range(len(accuracies)))
    # Set x-axis labels to correspond to the percentage of weights remaining after pruning
    pruning_percentages = [round(100 * (0.3 ** i),1) for i in pruning_levels]
    plt.xticks(pruning_levels, [f"{p}%" for p in pruning_percentages])
    #plt.plot(pruning_levels, accuracies, marker='o', label=hyperparam_str)
    if 'lt' in hyperparam_str:
        plt.plot(pruning_levels, accuracies, marker='o', label=hyperparam_str, color='black')
    else:
        plt.plot(pruning_levels, accuracies, marker='o', label=hyperparam_str)

plt.xlabel('Pruning Level (Weights Remaining)')
plt.ylabel('Best Test Accuracy (%)')
plt.legend()
plt.grid(True)

# Save the figure with a specified filename and desired resolution (DPI)
plt.savefig('pruning_accuracies_adam_02_fc1_fashionmnist.png', dpi=300)
# Optionally, display the plot
# plt.show()
