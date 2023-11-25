import matplotlib.pyplot as plt
import re
import os

# Define the directories for Bernoulli and LT files
bernoulli_dir = 'TEST_lt_resnet_rewind_bernoulli'
lt_dir = 'TEST_lthrewind_resnet18'

# Define the regex patterns to match the filenames and extract hyperparameters
bernoulli_pattern = r'bernoulli_kl(\S+)_datacifar10_archresnet18_percent(0\.7|0\.5)_lrp(\S+)_ntbernoulli_opt(SGD|Adam)_initialrewind.out'
lt_pattern = r'lt_initialrewind_datacifar10_archresnet18_percent(0\.7|0\.5)_opt(Adam|SGD)_rewind5.out'

# Pattern to find accuracy in the file content
accuracy_pattern = r'Best test Accuracy: (\d+\.\d+)%'

# Function to read accuracies from a file
def read_accuracies_from_file(file_path):
    with open(file_path, 'r') as file:
        log_contents = file.read()
    return [float(match) for match in re.findall(accuracy_pattern, log_contents)]

# Function to extract accuracies from log files
def extract_accuracies(log_directory, file_pattern, percent, opt):
    accuracies_dict = {}
    for filename in os.listdir(log_directory):
        if 'rewind10' in filename or 'rewind20' in filename or 'percent0.2' in filename:
            continue

        hyperparams_match = re.search(file_pattern, filename)
        if hyperparams_match:
            file_percent, file_opt = (hyperparams_match.group(1), hyperparams_match.group(2)) if 'lt_initialrewind' in filename else (hyperparams_match.group(2), hyperparams_match.group(4))
            if file_percent != percent or file_opt.lower() != opt.lower():
                continue

            key = 'lt' if 'lt_initialrewind' in filename else f"bernoulli_{hyperparams_match.group(1)}_lrp{hyperparams_match.group(3)}"
            accuracies = read_accuracies_from_file(os.path.join(log_directory, filename))
            accuracies_dict[key] = accuracies
    return accuracies_dict

# Function to plot data
def plot_data(bernoulli_data, lt_data, percent, opt):
    plt.figure(figsize=(10, 8))
    plt.title(f'Pruning Percent: {percent}, Optimizer: {opt}')

    # Plot Bernoulli data
    for key, accuracies in bernoulli_data.items():
        pruning_levels = list(range(len(accuracies)))
        plt.plot(pruning_levels, accuracies, marker='o', label=key)

    # Plot LT data
    if 'lt' in lt_data:
        accuracies = lt_data['lt']
        pruning_levels = list(range(len(accuracies)))
        plt.plot(pruning_levels, accuracies, marker='x', linestyle='--', label='LT')

    plt.xlabel('Pruning Level (Iterations)')
    plt.ylabel('Best Test Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the figure with a specified filename
    filename = f'pruning_accuracies_{percent}_{opt}_cifar10_resnet18.png'
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f'Saved figure {filename}')

# Process files and plot for each combination of percent and opt
for percent, opt in [('0.7', 'Adam'), ('0.7', 'SGD'), ('0.5', 'Adam'), ('0.5', 'SGD')]:
    bernoulli_data = extract_accuracies(bernoulli_dir, bernoulli_pattern, percent, opt)
    lt_data = extract_accuracies(lt_dir, lt_pattern, percent, opt)

    plot_data(bernoulli_data, lt_data, percent, opt)
