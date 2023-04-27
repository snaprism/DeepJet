#!/bin/bash

echo "Setting up container."

source $HOME/work/repositories/DeepJetCore/docker_env.sh
cd $HOME/work/repositories/DeepJet
source env.sh

export PYTHONPATH=/net/scratch_cms3a/ajung/python3.6.8/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=/net/scratch_cms3a/ajung/python3.6.8/lib64/python3.6/site-packages:$PYTHONPATH
export PATH=/net/scratch_cms3a/ajung/python3.6.8/bin:$PATH

source lxportal/execute_payload.sh
