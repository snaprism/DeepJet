#!/bin/bash

#export job_id=$(uuidgen)
export job_id=$$

nvidia-smi
echo "Which GPU do you want to use? 0, 1, 2 or 3?"
read gpu
echo "Using GPU ${gpu}."
export CUDA_VISIBLE_DEVICES=${gpu}

echo "A log will be saved here: /net/scratch_cms3a/ajung/deepjet/logs/output_$job_id.txt"

apptainer exec --nv --bind=$HOME,/net /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest /bin/bash $HOME/work/repositories/DeepJet/lxportal/setup_container.sh &>> /net/scratch_cms3a/ajung/deepjet/logs/output_$job_id.txt &
