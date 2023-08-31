import torch
import numpy as np 

# Function to make an empty mask of the same size as the model weights 
def make_mask(model):
    mask = []
    for param in model.parameters(): 
        mask.append(torch.ones_like(param.data))
    return mask

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

def reset_all_weights(model):
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)



def count_nonzero(model, mask):
    nonzero_model = 0
    nonzero_mask = 0
    total = 0
    mask_count = 0

    for i, param in enumerate(model.parameters()):
        # Calculate non-zeros in the model parameters and the mask
        nonzero_model += torch.count_nonzero(param.data)
        nonzero_mask += torch.count_nonzero(mask[i])

        # Calculate total elements in the model parameters and the mask
        total += param.data.numel()
        mask_count += mask[i].numel()
    ##print a statement that prints the percentage of nonzero weights in the model and the mask
    print(f"Non-zero model percentage: {nonzero_model / total * 100}%, Non-zero mask percentage: {nonzero_mask / mask_count * 100}%")

# Function for Training
def train(model, mask, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()                
                    
        step=0
        # Freezing Pruned weights by making their gradients Zero
        for param in model.parameters():
            param.grad.data = param.grad.data*mask[step]
            step+=1
        optimizer.step()
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    #test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        #test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def original_initialization(model, mask_temp, initial_state_dict): 
    step = 0
    for name, param in model.named_parameters(): 
        param.data = mask_temp[step] * initial_state_dict[name]
        step = step + 1
    return