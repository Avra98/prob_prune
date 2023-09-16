python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.1  --arch_type="resnet18"  --noise_step=50  --gpu=1  --dataset="cifar10"  --kl=1e-4 --prior=-4 --initial_p="last" --noise_batch=1024

python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.1  --arch_type="resnet18"  --noise_step=50  --gpu=1  --dataset="cifar10"  --kl=1e-7 --prior=-4 --initial_p="last" --noise_batch=1024

python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.1  --arch_type="resnet18"  --noise_step=50  --gpu=1  --dataset="cifar10"  --kl=1e-4 --prior=-2 --initial_p="last" --noise_batch=1024

python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.1  --arch_type="resnet18"  --noise_step=50  --gpu=1  --dataset="cifar10"  --kl=1e-7 --prior=-2 --initial_p="last" --noise_batch=1024




python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.1  --arch_type="resnet18"  --noise_step=50  --gpu=1  --dataset="cifar10"  --kl=1e-5 --prior=-4 --initial_p="last" --noise_batch=1024

python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.1  --arch_type="resnet18"  --noise_step=50  --gpu=1  --dataset="cifar10"  --kl=1e-6 --prior=-4 --initial_p="last" --noise_batch=1024

python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.1  --arch_type="resnet18"  --noise_step=50  --gpu=1  --dataset="cifar10"  --kl=1e-5 --prior=-2 --initial_p="last" --noise_batch=1024

python3 -u prune_clean.py --prune_type="noise"  --noise_type="bernoulli"  --end_iter=30 --lr=0.1  --arch_type="resnet18"  --noise_step=50  --gpu=1  --dataset="cifar10"  --kl=1e-6 --prior=-2 --initial_p="last" --noise_batch=1024

python3 -u prune_clean.py  --end_iter=30 --lr=0.1  --arch_type="resnet18"   --gpu=2   --dataset="cifar10" 

