#!/bin/bash

echo "Executing payload."
echo "############################## END OF BASH LOG ##############################"

### training
export input_dir=/net/scratch_cms3a/ajung/deepjet/data/medium_sample/dataCollection.djcdc
export output_dir=/net/scratch_cms3a/ajung/deepjet/results/fgsm_new
export adv=_fgsm
python3 pytorch/train_DeepFlavour$adv.py $input_dir $output_dir


### prediction
export checkpoint_dir=/net/scratch_cms3a/ajung/deepjet/results/fgsm_new/checkpoint_best_loss.pth
export traindata_dir=/net/scratch_cms3a/ajung/deepjet/results/fgsm_new/trainsamples.djcdc
export sample_dir=one_sample_lxportal.txt
export output_dir=/net/scratch_cms3a/ajung/deepjet/results/fgsm_new/predict
python3 pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir

export checkpoint_dir=/net/scratch_cms3a/ajung/deepjet/results/fgsm_new/checkpoint_best_loss.pth
export traindata_dir=/net/scratch_cms3a/ajung/deepjet/results/fgsm_new/trainsamples.djcdc
export sample_dir=one_sample_lxportal.txt
export output_dir=/net/scratch_cms3a/ajung/deepjet/results/fgsm_new/predict_fgsm
python3 pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir -attack FGSM -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 1

# -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 5


### evaluation
#python3 scripts/plot_loss_lxportal.py
#python3 scripts/plot_roc_lxportal.py


### testing
#echo; export; echo; nvidia-smi; echo; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvcc --version
#python3 $HOME/work/sandbox/test.py
