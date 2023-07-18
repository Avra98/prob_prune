import torch

# Defining a 1D tensor with requires_grad=True
p = torch.tensor([1.0], requires_grad=True)
# Defining optimizer with respect to p
optimizer = torch.optim.SGD([p], lr=0.1)

for i in range(40000):


    optimizer.zero_grad()
    # Draw a sample from N(0, 1)
    eps = torch.randn(1)
    # Reparameterize to get sample from N(0, p)
    #sample = torch.sqrt(p+ 1e-4) * eps 
    sample = torch.exp(p)*eps 

    #sample = torch.normal(0,torch.exp(p))
    # Compute a dummy loss: square of the sample
    loss = (sample**2).sum()
    loss.backward()
    optimizer.step()
    if i%1000==0:
        print(loss,p)

print(p)
