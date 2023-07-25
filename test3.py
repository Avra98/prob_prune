import torch.nn.functional as F
import torch
# Bernoulli probabilities
probabilities = torch.tensor([0.2, 0.8])  # Probability of outcomes 0 and 1

# Compute logits
logits = torch.log(probabilities / (1 - probabilities))

# Sample from Gumbel(0, 1)
gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))

# Add Gumbel noise to logits
perturbed_logits = logits + gumbel_noise

# Pass perturbed logits through softmax to obtain differentiable approximation of Bernoulli
soft_samples = F.softmax(perturbed_logits, dim=-1)

print(soft_samples)

