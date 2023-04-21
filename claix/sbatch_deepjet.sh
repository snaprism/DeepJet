#!/usr/bin/zsh

#SBATCH --account=rwth1244
#SBATCH --gres=gpu:1
#SBATCH --job-name=DeepJet
#SBATCH --mail-user=alexander.jung@rwth-aachen.de
#SBATCH --mail-type=END
#SBATCH --mem=10G
#SBATCH --output=/hpcwork/pj214607/work/promotion/deepjet/logs/output_%J.txt
#SBATCH --time=0-48:00:00

###for training: #SBATCH --time=0-48:00:00, for prediction and evaluation: #SBATCH --time=0-00:20:00

module purge
module restore deepjet

apptainer exec --nv --bind=/hpcwork/rwth1244,/work/rwth1244,/home/rwth1244,$HOME,$WORK,$HPCWORK /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest /bin/bash $HOME/work/repositories/DeepJet/claix/setup_container.sh
