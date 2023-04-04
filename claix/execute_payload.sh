#!/bin/bash

###training
export input_dir=/hpcwork/rwth1244/pj214607/deepjet/data/medium_sample/dataCollection.djcdc
export output_dir=/hpcwork/pj214607/work/promotion/deepjet/results/nominal
export adv=
python3 pytorch/train_DeepFlavour$adv.py $input_dir $output_dir


###prediction
###export checkpoint_dir=/hpcwork/rwth1244/pj214607/deepjet/train_df_run2/models/nominal_seed_0/checkpoint_best_loss.pth
###export traindata_dir=/hpcwork/rwth1244/pj214607/deepjet/train_df_run2/models/nominal_seed_0/trainsamples.djcdc
###export sample_dir=one_sample_claix.txt
###export output_dir=/hpcwork/rwth1244/pj214607/test

###python3 pytorch/predict_pytorch.py DeepJet_Run2 $checkpoint_dir $traindata_dir $sample_dir $output_dir