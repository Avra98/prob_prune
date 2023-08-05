import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def l2_norm(samples):
    return torch.norm(samples)


def generate_noise_soft(logits,temp=0.5):
    gumbel1 = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)
    gumbel2 = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)
    #gumbel1 = torch.rand_like(logits).requires_grad_(False)
    #gumbel2 = torch.rand_like(logits).requires_grad_(False)
    #gumbel1 =  torch.normal(0.5, 0.05, size=logits.size(), requires_grad=False)
    #gumbel2 =  torch.normal(0.5, 0.05, size=logits.size(), requires_grad=False)
    
    numerator = torch.exp((logits + gumbel1)/temp)
    denominator = torch.exp((logits + gumbel1)/temp)  + torch.exp(((1 - logits) + gumbel2)/temp)
    #print(gumbel1,gumbel2)
    
    noise = numerator / denominator

    return noise


# def monte_carlo_optimization(num_runs=8100, num_steps=500, learning_rate=0.01, temp=0.3):
#     logits = torch.tensor([0.7, 0.7, 0.7], requires_grad=True)
#     optimizer = optim.Adam([logits], lr=learning_rate)
#     loss_curve = []

#     for step in range(num_steps):
#         total_samples = torch.zeros_like(logits)
#         for _ in range(num_runs):
#             samples = generate_noise_soft(torch.norm(logits, dim=-1, keepdim=True), temp=temp)
#             total_samples += samples
#         average_samples = total_samples / num_runs

#         loss = l2_norm(average_samples)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         loss_curve.append(loss.item())

#     return logits,loss_curve

# logits,loss_curve = monte_carlo_optimization(num_runs=100, num_steps=600, learning_rate=0.01, temp=0.5)
# print(logits)

# plt.plot(range(1, len(loss_curve) + 1), loss_curve)
# plt.xlabel('Iterations')
# plt.ylabel('Loss (L2 norm)')
# plt.title('Loss Curve during Monte Carlo Optimization')


# plt.savefig('loss_curve.png')



logits = torch.tensor([0.99])  # Logits for a Bernoulli distribution with two categories
num_runs = 1000
total_samples = torch.zeros_like(logits)
hist=[]
for _ in range(num_runs):
    samples = generate_noise_soft(logits, temp=0.12)
    #print(samples)
    total_samples += samples
    hist.append(samples.item())

## Plotting the histogram of samples
plt.hist(hist, bins=100)
## save the histogram
plt.savefig('histogram.png')

average_samples = total_samples / num_runs
print(average_samples)
