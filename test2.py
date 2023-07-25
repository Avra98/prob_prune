import torch
import torch.nn.functional as F

# Define the Gumbel-Softmax functions
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y

# Define a 1-dimensional tensor of logits (e.g., log-probabilities)
logits = torch.tensor([0.5, 10.0, 0.0])

# Compute the Gumbel-Softmax
result = gumbel_softmax(logits, temperature=1.0)

print(result)
