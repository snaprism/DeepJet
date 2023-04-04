#!/bin/bash

source /home/pj214607/work/repositories/DeepJetCore/docker_env.sh
cd /home/pj214607/work/repositories/DeepJet
source env.sh

export PYTHONPATH=/hpcwork/rwth1244/pj214607/python3.6.8/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=/hpcwork/rwth1244/pj214607/python3.6.8/lib64/python3.6/site-packages:$PYTHONPATH
export PATH=/hpcwork/rwth1244/pj214607/python3.6.8/bin:$PATH

source claix/execute_payload.sh
