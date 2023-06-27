#!/bin/bash


### training
export input_dir=/net/scratch_cms3a/ajung/deepjet/data/medium_sample/dataCollection.djcdc
export output_dir=/net/scratch_cms3a/ajung/deepjet/results/fgsm_epsilon_5
export adv=_fgsm
python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/pytorch/train_DeepFlavour$adv.py $input_dir $output_dir


### prediction
#export checkpoint_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/checkpoint_best_loss.pth
#export traindata_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/trainsamples.djcdc
#export sample_dir=/home/home1/institut_3a/ajung/work/repositories/DeepJet/one_sample_lxportal.txt
#export output_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/predict_fgsm
#python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir -attack FGSM -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 1

#export checkpoint_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/checkpoint_best_loss.pth
#export traindata_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/trainsamples.djcdc
#export sample_dir=/home/home1/institut_3a/ajung/work/repositories/DeepJet/one_sample_lxportal.txt
#export output_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/predict_pgd_1
#python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 1

#export checkpoint_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/checkpoint_best_loss.pth
#export traindata_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/trainsamples.djcdc
#export sample_dir=/home/home1/institut_3a/ajung/work/repositories/DeepJet/one_sample_lxportal.txt
#export output_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/predict_pgd_2
#python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 2

#export checkpoint_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/checkpoint_best_loss.pth
#export traindata_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/trainsamples.djcdc
#export sample_dir=/home/home1/institut_3a/ajung/work/repositories/DeepJet/one_sample_lxportal.txt
#export output_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/predict_pgd_3
#python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 3

#export checkpoint_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/checkpoint_best_loss.pth
#export traindata_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/trainsamples.djcdc
#export sample_dir=/home/home1/institut_3a/ajung/work/repositories/DeepJet/one_sample_lxportal.txt
#export output_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/predict_pgd_4
#python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 4

#export checkpoint_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/checkpoint_best_loss.pth
#export traindata_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/trainsamples.djcdc
#export sample_dir=/home/home1/institut_3a/ajung/work/repositories/DeepJet/one_sample_lxportal.txt
#export output_dir=/net/scratch_cms3a/ajung/deepjet/results/nominal/predict_pgd_5
#python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 5

# -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 5


### evaluation
#python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/scripts/plot_loss_lxportal.py
#python3 /home/home1/institut_3a/ajung/work/repositories/DeepJet/scripts/plot_roc_lxportal.py


### testing
#echo; export; echo; nvidia-smi; echo; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvcc --version
#python3 $HOME/work/sandbox/test.py
