import torch
import argparse
import matplotlib.pyplot as plt

# Parse arguments
parser = argparse.ArgumentParser(description="Matrix completion with variable depth factorization")
parser.add_argument("--depth", type=int, default=2, help="Number of factor matrices")
parser.add_argument("--iters", type=int, default=10000, help="Number of iterations")
parser.add_argument("--true_rank", type=int, default=5, help="Rank of the original matrix")
parser.add_argument("--out_dim", type=int, default=50, help="Dimensionality of the matrices")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--mask_ratio", type=float, default=0.5, help="Ratio of observed entries in the mask")
parser.add_argument("--show_every", type=int, default=2000, help="show after how many iters")
parser.add_argument("--wd", type=float, default=0.00001, help="weight-decay")
args = parser.parse_args()

torch.manual_seed(42)

# Dimensionality of the matrices
d = args.out_dim
r = args.true_rank

# Original low-rank matrix
U = torch.randn(d, r)
V = torch.randn(d, r)
X_orig = U @ V.t()

# Compute the nuclear norm of the original matrix
nuclear_norm_orig = torch.linalg.norm(X_orig, ord='nuc')

print(f'The nuclear norm of the original matrix is: {nuclear_norm_orig.item()}')

# Mask (randomly hide some of the entries)
M = (torch.rand(d, d) < args.mask_ratio).float()

# Factors (initialized randomly)
factors = [torch.nn.Parameter(torch.randn(d, d)) for _ in range(args.depth)]

# Optimizer
optimizer = torch.optim.SGD(factors, lr=args.lr)

# Regularization weight
a = args.wd 

# Loss, nuclear norm, and true reconstruction error histories
losses = []
nuclear_norms = []
rec_errors = []

# Training loop
for i in range(args.iters):
    # Zero gradients
    optimizer.zero_grad()
    
    # Current estimate
    X_hat = factors[0]
    for j in range(1, args.depth):
        X_hat = X_hat @ factors[j]
    
    # Loss: mean squared error of observed entries plus Frobenius norm of each factor
    loss = ((M * (X_hat - X_orig))**2).mean()
    for factor in factors:
        loss += a * factor.norm(p='fro')**2

    # Backward pass
    loss.backward()

    # Update step
    optimizer.step()
    
    # Compute the true reconstruction error
    rec_error = ((X_hat - X_orig)**2).mean().item()/torch.norm(X_orig, p='fro')
    
    if i % args.show_every == 0:
        print(f'Loss at step {i}: {loss.item()}')
        nuclear_norm = torch.linalg.norm(X_hat.detach(), ord="nuc").item()
        print(f'Nuclear norm at step {i}: {nuclear_norm}')
        print(f'Reconstruction error at step {i}: {rec_error}')
        print('-' * 50)  # prints a horizontal line
        losses.append(loss.item())
        nuclear_norms.append(nuclear_norm)
        rec_errors.append(rec_error)


# Final reconstruction
X_rec = factors[0]
for j in range(1, args.depth):
    X_rec = X_rec @ factors[j].detach()

# Check nuclear norm of X_rec
print(f'Final nuclear norm of X_rec: {torch.linalg.norm(X_rec, ord="nuc").item()}')

# Plot loss, nuclear norm, and true reconstruction error history
# Plot loss, nuclear norm, and true reconstruction error history
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title(f"Loss, depth={args.depth}, lr={args.lr}, true_rank={args.true_rank}, dim={args.out_dim}")
plt.subplot(1, 3, 2)
plt.plot(nuclear_norms)
plt.axhline(y=nuclear_norm_orig.item(), color='r', linestyle='--')  # Add horizontal line
plt.title("Nuclear Norm")
plt.subplot(1, 3, 3)
plt.plot(rec_errors)
plt.title("Reconstruction Error")
plt.savefig(f'loss_nuclear_norm_rec_error_depth_{args.depth}.png')
plt.show()


# Compute the singular values
U, S, V = torch.svd(X_rec)

# Compute the Lp norm
lp_norm = torch.norm(S, p=0.5).item()

# Plot the singular values
plt.figure(figsize=(5, 5))
plt.plot(S.cpu().detach().numpy())
plt.title("Singular Value Spectrum of the Final Reconstruction")
plt.figtext(0.5, 0.01, f"Depth = {args.depth}, Lp norm for p={0.5}: {lp_norm}", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

# Save the figure with a name that includes the depth value
plt.savefig(f'singular_values_depth_{args.depth}_p_{0.5}.png')
plt.show()


