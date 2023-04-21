#!/bin/bash

###training
export input_dir=/hpcwork/rwth1244/pj214607/deepjet/data/medium_sample/dataCollection.djcdc
export output_dir=/hpcwork/pj214607/work/promotion/deepjet/results/fgsm
export adv=_adv
python3 pytorch/train_DeepFlavour$adv.py $input_dir $output_dir


###prediction
###export checkpoint_dir=/hpcwork/pj214607/work/promotion/deepjet/results/nominal/checkpoint_best_loss.pth
###export traindata_dir=/hpcwork/pj214607/work/promotion/deepjet/results/nominal/trainsamples.djcdc
###export sample_dir=one_sample_claix.txt
###export output_dir=/hpcwork/pj214607/work/promotion/deepjet/results/nominal/predict_pgd_1

###python3 pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 1

### -attack PGD -att_magnitude 0.01 -restrict_impact 0.2 -pgd_loops 5


###evaluation
###python3 scripts/plot_loss_claix.py
###python3 scripts/plot_roc_claix.py