from definitions import *
import torch
import numpy as np

def apply_noise(sample, magn=1e-2, offset=[0], dev=torch.device("cuda"), restrict_impact=-1, var_group="glob"):
    if magn == 0:
        return sample

    seed = 0
    np.random.seed(seed)

    with torch.no_grad():
        if var_group == 'glob':
            noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),vars_per_candidate[var_group]))).to(dev)
        else:
            noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),cands_per_variable[var_group],vars_per_candidate[var_group]))).to(dev)
        xadv = sample + noise

        if var_group == 'glob':
            for i in range(vars_per_candidate['glob']):
                if i in integer_variables_by_candidate[var_group]:
                    xadv[:,i] = sample[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults = sample[:,i].cpu() == defaults_per_variable[var_group][i]
                    if torch.sum(defaults) != 0:
                        xadv[:,i][defaults] = sample[:,i][defaults]

                    if restrict_impact > 0:
                        difference = xadv[:,i] - sample[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(sample[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv[high_impact,i] = sample[high_impact,i] + allowed_perturbation[high_impact] * torch.sign(noise[high_impact,i])

        else:
            for j in range(cands_per_variable[var_group]):
                for i in range(vars_per_candidate[var_group]):
                    if i in integer_variables_by_candidate[var_group]:
                        xadv[:,j,i] = sample[:,j,i]
                    else:
                        defaults = sample[:,j,i].cpu() == defaults_per_variable[var_group][i]
                        if torch.sum(defaults) != 0:
                            xadv[:,j,i][defaults] = sample[:,j,i][defaults]

                        if restrict_impact > 0:
                            difference = xadv[:,j,i] - sample[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(sample[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv[high_impact,j,i] = sample[high_impact,j,i] + allowed_perturbation[high_impact] * torch.sign(noise[high_impact,j,i])       

        return xadv

def fgsm_attack(epsilon=1e-2,sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, dev=torch.device("cuda"), restrict_impact=-1, epsilon_factors=True, return_grad=False):
    if epsilon == 0:
        return sample
    
    if epsilon_factors:
        eps_glob = torch.from_numpy(np.load(epsilons_per_feature["glob"])).cuda()
        eps_cpf  = torch.from_numpy(np.transpose(np.load(epsilons_per_feature["cpf"]))).cuda()
        eps_npf  = torch.from_numpy(np.transpose(np.load(epsilons_per_feature["npf"]))).cuda()
        eps_vtx  = torch.from_numpy(np.transpose(np.load(epsilons_per_feature["vtx"]))).cuda()
    else:
        eps_glob = 1.0
        eps_cpf  = 1.0
        eps_npf  = 1.0
        eps_vtx  = 1.0
    eps_fac = {"glob": eps_glob, "cpf": eps_cpf, "npf": eps_npf, "vtx": eps_vtx}

    glob, cpf, npf, vtx = sample
    
    xadv_glob = glob.clone().detach()
    xadv_cpf  = cpf.clone().detach()
    xadv_npf  = npf.clone().detach()
    xadv_vtx  = vtx.clone().detach()

    xadv_glob.requires_grad = True
    xadv_cpf.requires_grad  = True
    xadv_npf.requires_grad  = True
    xadv_vtx.requires_grad  = True

    preds = thismodel(xadv_glob,xadv_cpf,xadv_npf,xadv_vtx)

    loss = thiscriterion(preds, targets)

    thismodel.zero_grad()
    loss.backward()

    with torch.no_grad():
        dx_glob = torch.sign(xadv_glob.grad.detach())
        dx_cpf  = torch.sign(xadv_cpf.grad.detach())
        dx_npf  = torch.sign(xadv_npf.grad.detach())
        dx_vtx  = torch.sign(xadv_vtx.grad.detach())

        xadv_glob += epsilon * epsilon_factors['glob'] * dx_glob
        xadv_cpf  += epsilon * epsilon_factors['cpf']  * dx_cpf
        xadv_npf  += epsilon * epsilon_factors['npf']  * dx_npf
        xadv_vtx  += epsilon * epsilon_factors['vtx']  * dx_vtx
        if reduced:
            for i in range(vars_per_candidate['glob']):
                if i in integer_variables_by_candidate['glob']: #don't change integer varibles
                    xadv_glob[:,i] = glob[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:,i].cpu() == defaults_per_variable['glob'][i]
                    if torch.sum(defaults_glob) != 0: #don't change default varibles
                        xadv_glob[:,i][defaults_glob] = glob[:,i][defaults_glob]

                    if restrict_impact > 0:
                        difference = xadv_glob[:,i] - glob[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(glob[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv_glob[high_impact,i] = glob[high_impact,i] + allowed_perturbation[high_impact] * dx_glob[high_impact,i]

            for j in range(cands_per_variable['cpf']):
                for i in range(vars_per_candidate['cpf']):
                    if i in integer_variables_by_candidate['cpf']:
                        xadv_cpf[:,j,i] = cpf[:,j,i]
                    else:
                        defaults_cpf = cpf[:,j,i].cpu() == defaults_per_variable['cpf'][i]
                        if torch.sum(defaults_cpf) != 0:
                            xadv_cpf[:,j,i][defaults_cpf] = cpf[:,j,i][defaults_cpf]

                        if restrict_impact > 0:
                            difference = xadv_cpf[:,j,i] - cpf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(cpf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_cpf[high_impact,j,i] = cpf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_cpf[high_impact,j,i]        

            for j in range(cands_per_variable['npf']):
                for i in range(vars_per_candidate['npf']):
                    if i in integer_variables_by_candidate['npf']:
                        xadv_npf[:,j,i] = npf[:,j,i]
                    else:
                        defaults_npf = npf[:,j,i].cpu() == defaults_per_variable['npf'][i]
                        if torch.sum(defaults_npf) != 0:
                            xadv_npf[:,j,i][defaults_npf] = npf[:,j,i][defaults_npf]

                        if restrict_impact > 0:
                            difference = xadv_npf[:,j,i] - npf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(npf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_npf[high_impact,j,i] = npf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_npf[high_impact,j,i]   

            for j in range(cands_per_variable['vtx']):
                for i in range(vars_per_candidate['vtx']):
                    if i in integer_variables_by_candidate['vtx']:
                        xadv_vtx[:,j,i] = vtx[:,j,i]
                    else:
                        defaults_vtx = vtx[:,j,i].cpu() == defaults_per_variable['vtx'][i]
                        if torch.sum(defaults_vtx) != 0:
                            xadv_vtx[:,j,i][defaults_vtx] = vtx[:,j,i][defaults_vtx]

                        if restrict_impact > 0:
                            difference = xadv_vtx[:,j,i] - vtx[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(vtx[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_vtx[high_impact,j,i] = vtx[high_impact,j,i] + allowed_perturbation[high_impact] * dx_vtx[high_impact,j,i]   
        if return_grad:
            return xadv_glob.detach(), xadv_cpf.detach(), xadv_npf.detach(), xadv_vtx.detach(), dx_glob, dx_cpf, dx_npf, dx_vtx
        else:
            return xadv_glob.detach(), xadv_cpf.detach(), xadv_npf.detach(), xadv_vtx.detach()

    
def pgd_attack(epsilon=1e-2, pgd_loops=-1, sample=None, targets=None, thismodel=None, thiscriterion=None, reduced=True, dev=torch.device("cuda"), restrict_impact=-1, epsilon_factors=True):
    if epsilon == 0:
        return sample
    
    if epsilon_factors:
        eps_glob = torch.from_numpy(np.load(epsilons_per_feature["glob"])).cuda()
        eps_cpf  = torch.from_numpy(np.transpose(np.load(epsilons_per_feature["cpf"]))).cuda()
        eps_npf  = torch.from_numpy(np.transpose(np.load(epsilons_per_feature["npf"]))).cuda()
        eps_vtx  = torch.from_numpy(np.transpose(np.load(epsilons_per_feature["vtx"]))).cuda()
    else:
        eps_glob = 1.0
        eps_cpf  = 1.0
        eps_npf  = 1.0
        eps_vtx  = 1.0
    eps_fac = {"glob": eps_glob, "cpf": eps_cpf, "npf": eps_npf, "vtx": eps_vtx}
    
    glob, cpf, npf, vtx = sample
    glob_adv = glob.clone()
    cpf_adv  = cpf.clone()
    npf_adv  = npf.clone()
    vtx_adv  = vtx.clone()
    
    dx_glob = 0
    dx_cpf  = 0
    dx_npf  = 0
    dx_vtx  = 0
    for k in range(pgd_loops):
        if False:
        #if k%2==0:
            glob_adv = apply_noise(glob_adv, magn=1e-2, offset=[0], dev=torch.device("cuda"), restrict_impact=-1, var_group="glob")
            cpf_adv  = apply_noise(cpf_adv, magn=1e-2, offset=[0], dev=torch.device("cuda"), restrict_impact=-1, var_group="cpf")
            npf_adv  = apply_noise(npf_adv, magn=1e-2, offset=[0], dev=torch.device("cuda"), restrict_impact=-1, var_group="npf")
            vtx_adv  = apply_noise(vtx_adv, magn=1e-2, offset=[0], dev=torch.device("cuda"), restrict_impact=-1, var_group="vtx")
        
        glob_adv, cpf_adv, npf_adv, vtx_adv, dx_glob_, dx_cpf_, dx_npf_, dx_vtx_ = fgsm_attack(epsilon=epsilon, sample=(glob_adv, cpf_adv, npf_adv, vtx_adv), targets=targets, thismodel=thismodel, thiscriterion=thiscriterion, reduced=True, dev=dev, restrict_impact=-1, epsilon_factors=eps_fac, return_grad=True)
        
        delta    = torch.clamp(glob_adv-glob, min=-eps_fac["glob"]*epsilon, max=eps_fac["glob"]*epsilon)
        glob_adv = glob + delta
        
        delta    = torch.clamp(cpf_adv-cpf, min=-eps_fac["cpf"]*epsilon, max=eps_fac["cpf"]*epsilon)
        cpf_adv  = cpf + delta
        
        delta    = torch.clamp(npf_adv-npf, min=-eps_fac["npf"]*epsilon, max=eps_fac["npf"]*epsilon)
        npf_adv  = npf + delta
        
        delta    = torch.clamp(vtx_adv-vtx, min=-eps_fac["vtx"]*epsilon, max=eps_fac["vtx"]*epsilon)
        vtx_adv  = vtx + delta
        
        dx_glob += dx_glob_
        dx_cpf  += dx_cpf_
        dx_npf  += dx_npf_
        dx_vtx  += dx_vtx_

    if reduced:
        for i in range(vars_per_candidate['glob']):
            if i in integer_variables_by_candidate['glob']: #don't change integer variables
                glob_adv[:,i] = glob[:,i]
            else: # non integer, but might have defaults that should be excluded from shift
                defaults_glob = glob[:,i].cpu() == defaults_per_variable['glob'][i]
                if torch.sum(defaults_glob) != 0: #don't change default varibles
                    glob_adv[:,i][defaults_glob] = glob[:,i][defaults_glob]

                if restrict_impact > 0:
                    difference = glob_adv[:,i] - glob[:,i]
                    allowed_perturbation = restrict_impact * torch.abs(glob[:,i])
                    high_impact = torch.abs(difference) > allowed_perturbation

                    if torch.sum(high_impact)!=0:
                        glob_adv[high_impact,i] = glob[high_impact,i] + allowed_perturbation[high_impact] * dx_glob[high_impact,i]

        for j in range(cands_per_variable['cpf']):
            for i in range(vars_per_candidate['cpf']):
                if i in integer_variables_by_candidate['cpf']:
                    cpf_adv[:,j,i] = cpf[:,j,i]
                else:
                    defaults_cpf = cpf[:,j,i].cpu() == defaults_per_variable['cpf'][i]
                    if torch.sum(defaults_cpf) != 0:
                        cpf_adv[:,j,i][defaults_cpf] = cpf[:,j,i][defaults_cpf]

                    if restrict_impact > 0:
                        difference = cpf_adv[:,j,i] - cpf[:,j,i]
                        allowed_perturbation = restrict_impact * torch.abs(cpf[:,j,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            cpf_adv[high_impact,j,i] = cpf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_cpf[high_impact,j,i]        

        for j in range(cands_per_variable['npf']):
            for i in range(vars_per_candidate['npf']):
                if i in integer_variables_by_candidate['npf']:
                    npf_adv[:,j,i] = npf[:,j,i]
                else:
                    defaults_npf = npf[:,j,i].cpu() == defaults_per_variable['npf'][i]
                    if torch.sum(defaults_npf) != 0:
                        npf_adv[:,j,i][defaults_npf] = npf[:,j,i][defaults_npf]

                    if restrict_impact > 0:
                        difference = npf_adv[:,j,i] - npf[:,j,i]
                        allowed_perturbation = restrict_impact * torch.abs(npf[:,j,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            npf_adv[high_impact,j,i] = npf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_npf[high_impact,j,i]   

        for j in range(cands_per_variable['vtx']):
            for i in range(vars_per_candidate['vtx']):
                if i in integer_variables_by_candidate['vtx']:
                    vtx_adv[:,j,i] = vtx[:,j,i]
                else:
                    defaults_vtx = vtx[:,j,i].cpu() == defaults_per_variable['vtx'][i]
                    if torch.sum(defaults_vtx) != 0:
                        vtx_adv[:,j,i][defaults_vtx] = vtx[:,j,i][defaults_vtx]

                    if restrict_impact > 0:
                        difference = vtx_adv[:,j,i] - vtx[:,j,i]
                        allowed_perturbation = restrict_impact * torch.abs(vtx[:,j,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            vtx_adv[high_impact,j,i] = vtx[high_impact,j,i] + allowed_perturbation[high_impact] * dx_vtx[high_impact,j,i]   
    
    return glob_adv, cpf_adv, npf_adv, vtx_adv
