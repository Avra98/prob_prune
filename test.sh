#!/bin/bash

FILE="./Adam"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 2 3)
LEN=${#gpu_arr[@]}

for arch_type in resnet18
do
	for dataset in cifar10 fashionmnist
	do
		for percent in 0.5 0.8
		do
			for kl in 1e-5 1e-6 1e-7
			do
		    	for prior in -4 -2 0
		    	do
			    	python3 -u prune_clean.py --lr 1e-3 --momentum 0.0 --weight_decay 1e-4 --lr_p 1e-2 --prune_type="noise"  --noise_type="bernoulli" \
			    	 --end_iter=100 --arch_type $arch_type   --noise_step=50 --prune_percent $percent \
			    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset --kl $kl --optimizer Adam \
			    	 --prior $prior >> "$FILE"/"bernoulli_kl{$kl}_data{$dataset}_arch{$arch_type}_prior{$prior}_percent{$percent}.out" &	
			    	COUNTER=$((COUNTER + 1))
			    done

			    wait
		    done

		    python3 -u prune_clean.py --lr 1e-3 --momentum 0.0 --weight_decay 1e-4 --prune_type="lt" \
		    	 --end_iter=100 --arch_type $arch_type --prune_percent $percent \
		    	 --gpu ${gpu_arr[$((COUNTER%LEN))]} --dataset $dataset --optimizer Adam \
		    	  >> "$FILE"/"lt_data{$dataset}_arch{$arch_type}_percent{$percent}.out" &
		   	COUNTER=$((COUNTER + 1))

    	done
    done
done