python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.05  --arch_type="lenet5"   --noise_step=30  --gpu=5   --dataset="fashionmnist" --kl --prior=-2
python3 -u prune_clean.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=30 --lr=0.05  --arch_type="lenet5"   --noise_step=30  --gpu=5   --dataset="fashionmnist" --kl
python3 -u prune_clean.py  --end_iter=30 --lr=0.05  --arch_type="lenet5"   --gpu=5   --dataset="fashionmnist" 