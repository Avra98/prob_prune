#python3 -u prune.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=20 --lr=0.05  --arch_type="fc1" --kl="yes" --noise_step=20  --gpu=0 &
#python3 -u prune.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=20 --lr=0.05  --arch_type="fc1" --dataset="fashionmnist" --kl="yes" --noise_step=20  --gpu=1 &
#python3 -u prune.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=20 --lr=0.05  --arch_type="lenet5" --kl="yes" --noise_step=20  --gpu=2 &
#python3 -u prune.py --prune_type="noise"  --noise_type="gaussian"  --end_iter=20 --lr=0.05  --arch_type="lenet5" --dataset="fashionmnist" --kl="yes" --noise_step=20  --gpu=4 


python3 -u prune.py   --end_iter=20 --lr=0.01  --arch_type="resnet18" --dataset="cifar10"  --gpu=4 --initial="rewind" --rewind_iter=10
python3 -u prune.py   --end_iter=20 --lr=0.01  --arch_type="resnet18" --dataset="cifar10"  --gpu=4 --initial="last" 
python3 -u prune.py   --end_iter=20 --lr=0.01  --arch_type="resnet18" --dataset="cifar10"  --gpu=4 --initial="original" 