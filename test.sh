#!/bin/bash

FILE="./TEST"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 1 2 3)
LEN=${#gpu_arr[@]}

for arch_type in fc1 lenet5 vgg16
do
	for dataset in fashionmnist cifar10 cifar100
	do
		for percent in 0.5 0.8
		do
			for kl in 1e-5 1e-6 1e-7
			do
		    	    for prior in -4 -2 0
		    	    do

			    	python3 -u prune_clean.py --lr 1e-2 --momentum 0.0 --weight_decay 3e-4 --lr_p 1e-2 --prune_type="noise"  --noise_type="bernoulli" \
			    	 --end_iter=100 --arch_type $arch_type   --noise_step=100 --prune_percent $percent \
			    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset --kl $kl --optimizer Adam --prune_iterations 8 \
			    	 --prior $prior >> "$FILE"/"bernoulli_kl{$kl}_data{$dataset}_arch{$arch_type}_prior{$prior}_percent{$percent}.out" &	
			    	COUNTER=$((COUNTER + 1))
			    	
			    	if [ $((COUNTER%8)) -eq 0 ]
			    	then
			    	    wait
			    	fi

			    done

		    	python3 -u prune_clean.py --lr 1e-2 --momentum 0.0 --weight_decay 3e-4 --lr_p 1e-2 --prune_type="noise"  --noise_type="gaussian" \
		    	 --end_iter=100 --arch_type $arch_type   --noise_step=100 --prune_percent $percent \
		    	 --gpu ${gpu_arr[$((COUNTER%LEN))]}   --dataset $dataset --kl $kl --optimizer Adam --prune_iterations 8 \
		    	  >> "$FILE"/"gauss_kl{$kl}_data{$dataset}_arch{$arch_type}_percent{$percent}.out" &	
		    	COUNTER=$((COUNTER + 1))
		        if [ $((COUNTER%8)) -eq 0 ]
			then
			    wait
			fi

		    done

		    python3 -u prune_clean.py --lr 1e-2 --momentum 0.0 --weight_decay 3e-4 --prune_type="lt" \
		    	 --end_iter=100 --arch_type $arch_type --prune_percent $percent --prune_iterations 8 \
		    	 --gpu ${gpu_arr[$((COUNTER%LEN))]} --dataset $dataset --optimizer Adam \
		    	  >> "$FILE"/"lt_data{$dataset}_arch{$arch_type}_percent{$percent}.out" &
		   	COUNTER=$((COUNTER + 1))
		   	
		    if [ $((COUNTER%8)) -eq 0 ]
	 	    then
			wait
	            fi
		    python3 -u prune_clean.py --lr 1e-2 --momentum 0.0 --weight_decay 3e-4 --prune_type="noise_pac" \
		    	 --end_iter=100 --arch_type $arch_type --prune_percent $percent --prune_iterations 8 \
		    	 --gpu ${gpu_arr[$((COUNTER%LEN))]} --dataset $dataset --optimizer Adam \
		    	  >> "$FILE"/"pac_data{$dataset}_arch{$arch_type}_percent{$percent}.out" &
		   	COUNTER=$((COUNTER + 1))
		   	
		    if [ $((COUNTER%8)) -eq 0 ]
		    then
			wait
		    fi

    	done
    done
done
