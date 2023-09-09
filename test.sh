#!/bin/bash

FILE="./Adam_rerun"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 1 2 3 4)
LEN=${#gpu_arr[@]}

for arch_type in fc1
do
	for dataset in mnist cifar10 fashionmnist
	do
		for percent in 0.5 0.8
		do
			for kl in 1e-5 1e-6
			do
		    	for prior in -4 -2 
		    	do
			    	python3 -u prune_clean.py --lr 1e-3 --momentum 0.0 --weight_decay 1e-4 --lr_p 1e-2 --prune_type="noise"  --noise_type="bernoulli" \
			    	 --end_iter=100 --arch_type $arch_type   --noise_step=50 --prune_percent $percent \
			    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset --kl $kl --optimizer Adam \
			    	 --prior $prior >> "$FILE"/"bernoulli_kl{$kl}_data{$dataset}_arch{$arch_type}_prior{$prior}_percent{$percent}.out" &	
			    	COUNTER=$((COUNTER + 1))
			    done
		    done

		    python3 -u prune_clean.py --lr 1e-3 --momentum 0.0 --weight_decay 1e-4 --prune_type="lt" \
		    	 --end_iter=100 --arch_type $arch_type --prune_percent $percent \
		    	 --gpu ${gpu_arr[$((COUNTER%LEN))]} --dataset $dataset --optimizer Adam \
		    	  >> "$FILE"/"lt_data{$dataset}_arch{$arch_type}_percent{$percent}.out" &
		   	COUNTER=$((COUNTER + 1))

		   	wait
    	done
    done
done

FILE="./SGD_lr_large"
mkdir -p $FILE

for arch_type in lenet5 fc1
do
	for dataset in mnist cifar10 fashionmnist
	do
		for percent in 0.5 0.8
		do
			for kl in 1e-5 1e-6
			do
		    	for prior in -4 -2 
		    	do
			    	python3 -u prune_clean.py --lr 1e-2 --momentum 0.9 --weight_decay 1e-4 --lr_p 1e-2 --prune_type="noise"  --noise_type="bernoulli" \
			    	 --end_iter=100 --arch_type $arch_type   --noise_step=50 --prune_percent $percent \
			    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset --kl $kl --optimizer SGD \
			    	 --prior $prior >> "$FILE"/"bernoulli_kl{$kl}_data{$dataset}_arch{$arch_type}_prior{$prior}_percent{$percent}.out" &	
			    	COUNTER=$((COUNTER + 1))
			    done
		    done

		    python3 -u prune_clean.py --lr 1e-2 --momentum 0.9 --weight_decay 1e-4 --prune_type="lt" \
		    	 --end_iter=100 --arch_type $arch_type --prune_percent $percent \
		    	 --gpu ${gpu_arr[$((COUNTER%LEN))]} --dataset $dataset --optimizer SGD \
		    	  >> "$FILE"/"lt_data{$dataset}_arch{$arch_type}_percent{$percent}.out" &
		   	COUNTER=$((COUNTER + 1))

		   	wait
    	done
    done
done