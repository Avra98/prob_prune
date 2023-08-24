#python3 -u prune.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=20 --lr=0.05  --arch_type="fc1" --kl="yes" --noise_step=20  --gpu=0 &
#python3 -u prune.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=20 --lr=0.05  --arch_type="fc1" --dataset="fashionmnist" --kl="yes" --noise_step=20  --gpu=1 &
#python3 -u prune.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=20 --lr=0.05  --arch_type="lenet5" --kl="yes" --noise_step=20  --gpu=2 &
#python3 -u prune.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=20 --lr=0.05  --arch_type="lenet5" --dataset="fashionmnist" --kl="yes" --noise_step=20  --gpu=4 


# python3 -u prune.py   --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=0 --initial="rewind" --rewind_iter=10  --prune_iterations=6 
# python3 -u prune.py   --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=1 --initial="last"   --prune_iterations=6 
# python3 -u prune.py   --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=2 --initial="original"   --prune_iterations=6 

# python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=3  --initial="original" --prune_type="noise" --noise_type="gaussian" --kl="yes" --noise_step=20  --prune_iterations=6 

# python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=5  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="no" --noise_step=20  --prune_iterations=6 

# python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=6  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="yes" --noise_step=20  --prune_iterations=6 --prior=0.2
# python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=7  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="yes" --noise_step=20  --prune_iterations=6 --prior=-0.2

# python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=0  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="yes" --noise_step=20  --prune_iterations=6 --prior=1.0
# python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=2  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="yes" --noise_step=20  --prune_iterations=6 --prior=-1.0


#python3 -u prune.py   --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=0 --initial="rewind" --rewind_iter=10  --prune_iterations=6 
python3 -u prune.py   --end_iter=15 --lr=0.1  --arch_type="alexnet" --dataset="fashionmnist"  --gpu=1 --initial="last"   --prune_iterations=6 
#python3 -u prune.py   --end_iter=15 --lr=0.1  --arch_type="resnet18" --dataset="cifar10"  --gpu=2 --initial="original"   --prune_iterations=6 

python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="alexnet" --dataset="fashionmnist"  --gpu=3  --initial="original" --prune_type="noise" --noise_type="gaussian" --kl="yes" --noise_step=20  --prune_iterations=6 

python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="alexnet" --dataset="fashionmnist"  --gpu=5  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="no" --noise_step=20  --prune_iterations=6 

python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="alexnet" --dataset="fashionmnist"  --gpu=6  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="yes" --noise_step=20  --prune_iterations=6 --prior=0.2
python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="alexnet" --dataset="fashionmnist"  --gpu=7  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="yes" --noise_step=20  --prune_iterations=6 --prior=-0.2

python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="alexnet" --dataset="fashionmnist"  --gpu=0  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="yes" --noise_step=20  --prune_iterations=6 --prior=1.0
python3 -u prune.py  --end_iter=15 --lr=0.1  --arch_type="alexnet" --dataset="fashionmnist"  --gpu=2  --initial="original" --prune_type="noise" --noise_type="bernoulli" --kl="yes" --noise_step=20  --prune_iterations=6 --prior=-1.0