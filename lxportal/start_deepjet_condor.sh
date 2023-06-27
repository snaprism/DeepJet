#!/bin/bash

apptainer exec --nv --bind=/home/home1/institut_3a/ajung,/net /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest /bin/bash /home/home1/institut_3a/ajung/work/repositories/DeepJet/lxportal/setup_container_condor.sh