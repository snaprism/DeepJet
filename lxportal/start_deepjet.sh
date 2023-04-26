#!/biin/bash

nvidia-smi
echo "Which GPU do you want to use? 0, 1, 2 or 3?"
read gpu
echo "Using GPU ${gpu}."
export CUDA_VISIBLE_DEVICES=${gpu}

apptainer exec --nv --bind=$HOME,/net /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest /bin/bash $HOME/work/repositories/DeepJet/lxportal/setup_container.sh