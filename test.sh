#!/bin/bash

FILE="./logs"
mkdir -p $FILE

COUNTER=0
gpu_arr=(2 6 7)
LEN=${#gpu_arr[@]}

for arch_type in lenet5 fc1
do
	for dataset in mnist fashionmnist
	do
		for kl in 0.0 1e-4 1e-5 1e-6
		do
	    	python3 -u prune_clean.py --lr 1e-2 --momentum 0.9 --prune_type="noise"  --noise_type="bernoulli" \
	    	 --end_iter=50 --arch_type $arch_type   --noise_step=30 \
	    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset --kl $kl \
	    	 --prior=-2 >> "$FILE"/"bernoulli_kl{$kl}_data{$dataset}_arch{$arch_type}.out" &	
	    	COUNTER=$((COUNTER + 1))

	    	python3 -u prune_clean.py --lr 1e-2 --momentum 0.9 --prune_type="noise"  --noise_type="gaussian" \
	    	 --end_iter=50 --arch_type $arch_type   --noise_step=30 \
	    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset \
	    	 --kl $kl >> "$FILE"/"gaussian_kl{$kl}_data{$dataset}_arch{$arch_type}.out" &
	    	COUNTER=$((COUNTER + 1))
	    done

	    python3 -u prune_clean.py --lr 1e-2 --momentum 0.9 --prune_type="lt" \
	    	 --end_iter=50 --arch_type $arch_type   --noise_step=30 \
	    	 --gpu ${gpu_arr[$((COUNTER%LEN))]} --dataset $dataset \
	    	  >> "$FILE"/"lt_data{$dataset}_arch{$arch_type}.out" &
	   	COUNTER=$((COUNTER + 1))

    	python3 -u prune_clean.py --lr 1e-2 --momentum 0.9 --prune_type="noise_pac" --noise_type="gaussian" \
    	 --end_iter=50 --arch_type $arch_type   --noise_step=30 \
    	 --gpu 0   --dataset $dataset --kl 1.0 >> "$FILE"/"pac_data{$dataset}_arch{$arch_type}.out"  \
    	 --prior=-2 
    	
    	wait
    done
done