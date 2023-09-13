#!/bin/bash
outfolder="jobs/Adam"
mkdir -p -- ${outfolder}

CmdsArray=()
for arch_type in fc1 lenet5 vgg16 resnet18
do
	for dataset in cifar10
	do
		for percent in 0.5
		do
			for kl in 1e-5 1e-6 1e-7
			do
			    	for prior in -4 -2 0
			    	do

				    	cmd="python3 -u prune_clean.py --lr 1e-3 --momentum 0.0 --weight_decay 3e-4 --lr_p 1e-2 --prune_type="noise"  --noise_type="bernoulli" \
					    	 --end_iter=100 --arch_type $arch_type   --noise_step=100 --prune_percent $percent \
					    	 --dataset $dataset --kl $kl --optimizer Adam --prune_iterations 8 \
					    	 --prior $prior >> "$outfolder"/"bernoulli_kl{$kl}_data{$dataset}_arch{$arch_type}_prior{$prior}_percent{$percent}.out""	
					CmdsArray+=("$cmd")
			    	done

				cmd="python3 -u prune_clean.py --lr 1e-3 --momentum 0.0 --weight_decay 3e-4 --lr_p 1e-2 --prune_type="noise"  --noise_type="bernoulli" \
					 --end_iter=100 --arch_type $arch_type   --noise_step=100 --prune_percent $percent \
					 --dataset $dataset --kl $kl --optimizer Adam --prune_iterations 8 \
					 --prior $prior >> "$outfolder"/"gauss_kl{$kl}_data{$dataset}_arch{$arch_type}_prior{$prior}_percent{$percent}.out""	
				CmdsArray+=("$cmd")
		    done

			cmd="python3 -u prune_clean.py --lr 1e-3 --momentum 0.0 --weight_decay 3e-4 --prune_type="lt" \
				 --end_iter=100 --arch_type $arch_type --noise_step=100 --prune_percent $percent \
				 --dataset $dataset --optimizer Adam --prune_iterations 8 \
				  >> "$outfolder"/"lt_data{$dataset}_arch{$arch_type}_percent{$percent}.out""
			CmdsArray+=("$cmd")

			cmd="python3 -u prune_clean.py --lr 1e-3 --momentum 0.0 --weight_decay 3e-4 --prune_type="noise_pac" \
				 --end_iter=100 --arch_type $arch_type --noise_step=100 --prune_percent $percent \
				 --dataset $dataset --optimizer Adam --prune_iterations 8 \
				  >> "$outfolder"/"pac_data{$dataset}_arch{$arch_type}_percent{$percent}.out""
    			CmdsArray+=("$cmd")
    		done
    done
done