#!/bin/bash

FILE="./logs"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 1 2 3 5 6 7)
LEN=${#gpu_arr[@]}

for arch_type in lenet5 fc1
do
	for dataset in mnist fashionmnist
	do
		for percent in 0.25 0.5 0.8
		do
			for kl in 0.0 1e-4 1e-5 1e-6
			do
		    	for prior in -6 -4 -2 2
		    	do
			    	python3 -u prune_clean.py --lr 1e-2 --lr_p 1e-1 --momentum 0.0 --prune_type="noise"  --noise_type="bernoulli" \
			    	 --end_iter=50 --arch_type $arch_type   --noise_step=30 --prune_percent $percent \
			    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset --kl $kl \
			    	 --prior $prior >> "$FILE"/"bernoulli_kl{$kl}_data{$dataset}_arch{$arch_type}_prior{$prior}_percent{$percent}.out" &	
			    	COUNTER=$((COUNTER + 1))
			    done

		    	python3 -u prune_clean.py --lr 1e-2 --lr_p 1e-1 --momentum 0.0 --prune_type="noise"  --noise_type="gaussian" \
		    	 --end_iter=50 --arch_type $arch_type   --noise_step=30 --prune_percent $percent \
		    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset \
		    	 --kl $kl >> "$FILE"/"gaussian_kl{$kl}_data{$dataset}_arch{$arch_type}_percent{$percent}.out" &
		    	COUNTER=$((COUNTER + 1))
		    done

		    python3 -u prune_clean.py --lr 1e-2 --momentum 0.0 --prune_type="lt" \
		    	 --end_iter=50 --arch_type $arch_type   --noise_step=30 --prune_percent $percent \
		    	 --gpu ${gpu_arr[$((COUNTER%LEN))]} --dataset $dataset \
		    	  >> "$FILE"/"lt_data{$dataset}_arch{$arch_type}_percent{$percent}.out" 
		   	COUNTER=$((COUNTER + 1))
		   	
	    	wait
    	done
    done
done