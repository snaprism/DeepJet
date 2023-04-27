cands_per_variable = {
    'glob' : 1,
    'cpf' : 25,
    'npf' : 25,
#    'vtx' : 5,
    'vtx' : 4,
    #'pxl' : ,
}
vars_per_candidate = {
    'glob' : 15,
    'cpf' : 16,
#    'npf' : 8,
    'npf' : 6,
#    'vtx' : 14,
    'vtx' : 12,
    #'pxl' : ,
}
defaults_per_variable_before_prepro = {
    'glob' : [None,None,None,None,None,None,-999,-999,-999,-999,-999,-999,-999,None,None],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
    #'pxl' : ,
}
# lxplus
#epsilons_per_feature = {
#    'glob' : '/eos/user/a/aljung/DeepJet/Train_DF_Run2/auxiliary/new_global_epsilons.npy',
#    'cpf' : '/eos/user/a/aljung/DeepJet/Train_DF_Run2/auxiliary/new_cpf_epsilons.npy',
#    'npf' : '/eos/user/a/aljung/DeepJet/Train_DF_Run2/auxiliary/new_npf_epsilons.npy',
#    'vtx' : '/eos/user/a/aljung/DeepJet/Train_DF_Run2/auxiliary/new_vtx_epsilons.npy'
#}
# claix
#epsilons_per_feature = {
#    'glob' : '/hpcwork/rwth1244/pj214607/deepjet/auxiliary/new_global_epsilons.npy',
#    'cpf' : '/hpcwork/rwth1244/pj214607/deepjet/auxiliary/new_cpf_epsilons.npy',
#    'npf' : '/hpcwork/rwth1244/pj214607/deepjet/auxiliary/new_npf_epsilons.npy',
#    'vtx' : '/hpcwork/rwth1244/pj214607/deepjet/auxiliary/new_vtx_epsilons.npy'
#}
# lxportal
epsilons_per_feature = {
    'glob' : '/net/scratch_cms3a/ajung/deepjet/auxiliary/new_global_epsilons.npy',
    'cpf' : '/net/scratch_cms3a/ajung/deepjet/auxiliary/new_cpf_epsilons.npy',
    'npf' : '/net/scratch_cms3a/ajung/deepjet/auxiliary/new_npf_epsilons.npy',
    'vtx' : '/net/scratch_cms3a/ajung/deepjet/auxiliary/new_vtx_epsilons.npy'
}

defaults_per_variable = {
    'glob' : [0 for i in range(vars_per_candidate['glob'])],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
    #'pxl' : ,
}
integer_variables_by_candidate = {
    'glob' : [2,3,4,5,8,13,14],
    'cpf' : [12,13,14,15], # adding 14 because chi2 is an approximante integer
#    'npf' : [4],
    'npf' : [2],
#    'vtx' : [5],
    'vtx' : [3],
    #'pxl' : ,
}
