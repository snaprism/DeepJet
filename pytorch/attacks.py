from definitions import *
import numpy as np
import torch

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def apply_noise(
    sample,
    magn=1e-2,
    offset=[0],
    dev=torch.device("cuda"),
    restrict_impact=-1,
    var_group="glob",
):
    if magn == 0:
        return sample

    seed = 0
    np.random.seed(seed)

    with torch.no_grad():
        if var_group == "glob":
            noise = torch.Tensor(
                np.random.normal(
                    offset, magn, (len(sample), vars_per_candidate[var_group])
                )
            ).to(dev)
        else:
            noise = torch.Tensor(
                np.random.normal(
                    offset,
                    magn,
                    (
                        len(sample),
                        cands_per_variable[var_group],
                        vars_per_candidate[var_group],
                    ),
                )
            ).to(dev)
        xadv = sample + noise

        if var_group == "glob":
            for i in range(vars_per_candidate["glob"]):
                if i in integer_variables_by_candidate[var_group]:
                    xadv[:, i] = sample[:, i]
                else:  # non integer, but might have defaults that should be excluded from shift
                    defaults = sample[:, i].cpu() == defaults_per_variable[var_group][i]
                    if torch.sum(defaults) != 0:
                        xadv[:, i][defaults] = sample[:, i][defaults]

                    if restrict_impact > 0:
                        difference = xadv[:, i] - sample[:, i]
                        allowed_perturbation = restrict_impact * torch.abs(sample[:, i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact) != 0:
                            xadv[high_impact, i] = sample[
                                high_impact, i
                            ] + allowed_perturbation[high_impact] * torch.sign(
                                noise[high_impact, i]
                            )

        else:
            for j in range(cands_per_variable[var_group]):
                for i in range(vars_per_candidate[var_group]):
                    if i in integer_variables_by_candidate[var_group]:
                        xadv[:, j, i] = sample[:, j, i]
                    else:
                        defaults = (
                            sample[:, j, i].cpu() == defaults_per_variable[var_group][i]
                        )
                        if torch.sum(defaults) != 0:
                            xadv[:, j, i][defaults] = sample[:, j, i][defaults]

                        if restrict_impact > 0:
                            difference = xadv[:, j, i] - sample[:, j, i]
                            allowed_perturbation = restrict_impact * torch.abs(
                                sample[:, j, i]
                            )
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact) != 0:
                                xadv[high_impact, j, i] = sample[
                                    high_impact, j, i
                                ] + allowed_perturbation[high_impact] * torch.sign(
                                    noise[high_impact, j, i]
                                )

        return xadv


def fgsm_attack(
    epsilon=1e-2,
    sample=None,
    targets=None,
    thismodel=None,
    thiscriterion=None,
    reduced=True,
    dev=torch.device("cuda"),
    restrict_impact=-1,
    epsilon_factors=True,
    return_grad=False,
):
    if epsilon == 0:
        return sample

    if epsilon_factors:
        eps_glob = torch.from_numpy(np.load(epsilons_per_feature["glob"])).to(dev)
        eps_cpf = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["cpf"]))
        ).to(dev)
        eps_npf = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["npf"]))
        ).to(dev)
        eps_vtx = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["vtx"]))
        ).to(dev)
    else:
        eps_glob = 1.0
        eps_cpf = 1.0
        eps_npf = 1.0
        eps_vtx = 1.0
    eps_fac = {"glob": eps_glob, "cpf": eps_cpf, "npf": eps_npf, "vtx": eps_vtx}

    glob, cpf, npf, vtx = sample

    # xadv_glob = glob.clone().detach()
    # xadv_cpf  = cpf.clone().detach()
    # xadv_npf  = npf.clone().detach()
    # xadv_vtx  = vtx.clone().detach()
    xadv_glob = glob.clone().detach().to(dev)
    xadv_cpf = cpf.clone().detach().to(dev)
    xadv_npf = npf.clone().detach().to(dev)
    xadv_vtx = vtx.clone().detach().to(dev)

    xadv_glob.requires_grad = True
    xadv_cpf.requires_grad = True
    xadv_npf.requires_grad = True
    xadv_vtx.requires_grad = True

    preds = thismodel(xadv_glob, xadv_cpf, xadv_npf, xadv_vtx)

    loss = thiscriterion(preds, targets)

    thismodel.zero_grad()
    loss.backward()

    with torch.no_grad():
        # dx_glob = torch.sign(xadv_glob.grad.detach())
        # dx_cpf  = torch.sign(xadv_cpf.grad.detach())
        # dx_npf  = torch.sign(xadv_npf.grad.detach())
        # dx_vtx  = torch.sign(xadv_vtx.grad.detach())
        dx_glob = xadv_glob.grad.detach().sign()
        dx_cpf = xadv_cpf.grad.detach().sign()
        dx_npf = xadv_npf.grad.detach().sign()
        dx_vtx = xadv_vtx.grad.detach().sign()

        xadv_glob += epsilon * epsilon_factors["glob"] * dx_glob
        xadv_cpf += epsilon * epsilon_factors["cpf"] * dx_cpf
        xadv_npf += epsilon * epsilon_factors["npf"] * dx_npf
        xadv_vtx += epsilon * epsilon_factors["vtx"] * dx_vtx
        if reduced:
            for i in range(vars_per_candidate["glob"]):
                if (
                    i in integer_variables_by_candidate["glob"]
                ):  # don't change integer varibles
                    xadv_glob[:, i] = glob[:, i]
                else:  # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:, i].cpu() == defaults_per_variable["glob"][i]
                    if torch.sum(defaults_glob) != 0:  # don't change default varibles
                        xadv_glob[:, i][defaults_glob] = glob[:, i][defaults_glob]

                    if restrict_impact > 0:
                        difference = xadv_glob[:, i] - glob[:, i]
                        allowed_perturbation = restrict_impact * torch.abs(glob[:, i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact) != 0:
                            xadv_glob[high_impact, i] = (
                                glob[high_impact, i]
                                + allowed_perturbation[high_impact]
                                * dx_glob[high_impact, i]
                            )

            for j in range(cands_per_variable["cpf"]):
                for i in range(vars_per_candidate["cpf"]):
                    if i in integer_variables_by_candidate["cpf"]:
                        xadv_cpf[:, j, i] = cpf[:, j, i]
                    else:
                        defaults_cpf = (
                            cpf[:, j, i].cpu() == defaults_per_variable["cpf"][i]
                        )
                        if torch.sum(defaults_cpf) != 0:
                            xadv_cpf[:, j, i][defaults_cpf] = cpf[:, j, i][defaults_cpf]

                        if restrict_impact > 0:
                            difference = xadv_cpf[:, j, i] - cpf[:, j, i]
                            allowed_perturbation = restrict_impact * torch.abs(
                                cpf[:, j, i]
                            )
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact) != 0:
                                xadv_cpf[high_impact, j, i] = (
                                    cpf[high_impact, j, i]
                                    + allowed_perturbation[high_impact]
                                    * dx_cpf[high_impact, j, i]
                                )

            for j in range(cands_per_variable["npf"]):
                for i in range(vars_per_candidate["npf"]):
                    if i in integer_variables_by_candidate["npf"]:
                        xadv_npf[:, j, i] = npf[:, j, i]
                    else:
                        defaults_npf = (
                            npf[:, j, i].cpu() == defaults_per_variable["npf"][i]
                        )
                        if torch.sum(defaults_npf) != 0:
                            xadv_npf[:, j, i][defaults_npf] = npf[:, j, i][defaults_npf]

                        if restrict_impact > 0:
                            difference = xadv_npf[:, j, i] - npf[:, j, i]
                            allowed_perturbation = restrict_impact * torch.abs(
                                npf[:, j, i]
                            )
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact) != 0:
                                xadv_npf[high_impact, j, i] = (
                                    npf[high_impact, j, i]
                                    + allowed_perturbation[high_impact]
                                    * dx_npf[high_impact, j, i]
                                )

            for j in range(cands_per_variable["vtx"]):
                for i in range(vars_per_candidate["vtx"]):
                    if i in integer_variables_by_candidate["vtx"]:
                        xadv_vtx[:, j, i] = vtx[:, j, i]
                    else:
                        defaults_vtx = (
                            vtx[:, j, i].cpu() == defaults_per_variable["vtx"][i]
                        )
                        if torch.sum(defaults_vtx) != 0:
                            xadv_vtx[:, j, i][defaults_vtx] = vtx[:, j, i][defaults_vtx]

                        if restrict_impact > 0:
                            difference = xadv_vtx[:, j, i] - vtx[:, j, i]
                            allowed_perturbation = restrict_impact * torch.abs(
                                vtx[:, j, i]
                            )
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact) != 0:
                                xadv_vtx[high_impact, j, i] = (
                                    vtx[high_impact, j, i]
                                    + allowed_perturbation[high_impact]
                                    * dx_vtx[high_impact, j, i]
                                )
        if return_grad:
            return (
                xadv_glob.detach(),
                xadv_cpf.detach(),
                xadv_npf.detach(),
                xadv_vtx.detach(),
                dx_glob,
                dx_cpf,
                dx_npf,
                dx_vtx,
            )
        else:
            return (
                xadv_glob.detach(),
                xadv_cpf.detach(),
                xadv_npf.detach(),
                xadv_vtx.detach(),
            )


"""
def pgd_attack(
    epsilon=1e-2,
    pgd_loops=-1,
    sample=None,
    targets=None,
    thismodel=None,
    thiscriterion=None,
    reduced=True,
    dev=torch.device("cuda"),
    restrict_impact=-1,
    epsilon_factors=True,
    batch_index=None,
):
    if epsilon == 0:
        return sample

    if epsilon_factors:
        epsilon_glob = torch.from_numpy(np.load(epsilons_per_feature["glob"])).to(dev)
        epsilon_cpf = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["cpf"]))
        ).to(dev)
        epsilon_npf = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["npf"]))
        ).to(dev)
        epsilon_vtx = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["vtx"]))
        ).to(dev)
    else:
        epsilon_glob = 1.0
        epsilon_cpf = 1.0
        epsilon_npf = 1.0
        epsilon_vtx = 1.0

    glob, cpf, npf, vtx = sample

    adv_glob = glob.clone().detach().to(dev)
    adv_cpf = cpf.clone().detach().to(dev)
    adv_npf = npf.clone().detach().to(dev)
    adv_vtx = vtx.clone().detach().to(dev)

    dx_glob = 0
    dx_cpf = 0
    dx_npf = 0
    dx_vtx = 0
    for l in range(pgd_loops):
        adv_glob.requires_grad = True
        adv_cpf.requires_grad = True
        adv_npf.requires_grad = True
        adv_vtx.requires_grad = True

        prediction = thismodel(adv_glob, adv_cpf, adv_npf, adv_vtx)

        loss = thiscriterion(prediction, targets)

        thismodel.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_glob = adv_glob.grad.detach().sign()
            grad_cpf = adv_cpf.grad.detach().sign()
            grad_npf = adv_npf.grad.detach().sign()
            grad_vtx = adv_vtx.grad.detach().sign()

            adv_glob += epsilon * epsilon_glob * grad_glob
            adv_cpf += epsilon * epsilon_cpf * grad_cpf
            adv_npf += epsilon * epsilon_npf * grad_npf
            adv_vtx += epsilon * epsilon_vtx * grad_vtx

            delta_glob = torch.clamp(
                adv_glob - glob,
                min=-epsilon * epsilon_factors["glob"],
                max=epsilon * epsilon_factors["glob"],
            )
            delta_cpf = torch.clamp(
                adv_cpf - cpf,
                min=-epsilon * epsilon_factors["cpf"],
                max=epsilon * epsilon_factors["cpf"],
            )
            delta_npf = torch.clamp(
                adv_npf - npf,
                min=-epsilon * epsilon_factors["npf"],
                max=epsilon * epsilon_factors["npf"],
            )
            delta_vtx = torch.clamp(
                adv_vtx - vtx,
                min=-epsilon * epsilon_factors["vtx"],
                max=epsilon * epsilon_factors["vtx"],
            )

            adv_glob = glob + delta_glob
            adv_cpf = cpf + delta_cpf
            adv_npf = npf + delta_npf
            adv_vtx = vtx + delta_vtx

            dx_glob += grad_glob
            dx_cpf += grad_cpf
            dx_npf += grad_npf
            dx_vtx += grad_vtx

            np.save(
                f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/glob_{batch_index}_{l}.npy",
                adv_glob.detach().cpu().numpy(),
            )
            np.save(
                f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/cpf_{batch_index}_{l}.npy",
                adv_cpf.detach().cpu().numpy(),
            )
            np.save(
                f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/npf_{batch_index}_{l}.npy",
                adv_npf.detach().cpu().numpy(),
            )
            np.save(
                f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/vtx_{batch_index}_{l}.npy",
                adv_vtx.detach().cpu().numpy(),
            )

    with torch.no_grad():
        if reduced:
            for i in range(vars_per_candidate["glob"]):
                if (
                    i in integer_variables_by_candidate["glob"]
                ):  # don't change integer variables
                    adv_glob[:, i] = glob[:, i]
                else:  # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:, i].cpu() == defaults_per_variable["glob"][i]
                    if torch.sum(defaults_glob) != 0:  # don't change default varibles
                        adv_glob[:, i][defaults_glob] = glob[:, i][defaults_glob]

                    if restrict_impact > 0:
                        difference = adv_glob[:, i] - glob[:, i]
                        allowed_perturbation = restrict_impact * torch.abs(glob[:, i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact) != 0:
                            adv_glob[high_impact, i] = (
                                glob[high_impact, i]
                                + allowed_perturbation[high_impact]
                                * dx_glob[high_impact, i]
                            )

            for j in range(cands_per_variable["cpf"]):
                for i in range(vars_per_candidate["cpf"]):
                    if i in integer_variables_by_candidate["cpf"]:
                        adv_cpf[:, j, i] = cpf[:, j, i]
                    else:
                        defaults_cpf = (
                            cpf[:, j, i].cpu() == defaults_per_variable["cpf"][i]
                        )
                        if torch.sum(defaults_cpf) != 0:
                            adv_cpf[:, j, i][defaults_cpf] = cpf[:, j, i][defaults_cpf]

                        if restrict_impact > 0:
                            difference = adv_cpf[:, j, i] - cpf[:, j, i]
                            allowed_perturbation = restrict_impact * torch.abs(
                                cpf[:, j, i]
                            )
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact) != 0:
                                adv_cpf[high_impact, j, i] = (
                                    cpf[high_impact, j, i]
                                    + allowed_perturbation[high_impact]
                                    * dx_cpf[high_impact, j, i]
                                )

            for j in range(cands_per_variable["npf"]):
                for i in range(vars_per_candidate["npf"]):
                    if i in integer_variables_by_candidate["npf"]:
                        adv_npf[:, j, i] = npf[:, j, i]
                    else:
                        defaults_npf = (
                            npf[:, j, i].cpu() == defaults_per_variable["npf"][i]
                        )
                        if torch.sum(defaults_npf) != 0:
                            adv_npf[:, j, i][defaults_npf] = npf[:, j, i][defaults_npf]

                        if restrict_impact > 0:
                            difference = adv_npf[:, j, i] - npf[:, j, i]
                            allowed_perturbation = restrict_impact * torch.abs(
                                npf[:, j, i]
                            )
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact) != 0:
                                adv_npf[high_impact, j, i] = (
                                    npf[high_impact, j, i]
                                    + allowed_perturbation[high_impact]
                                    * dx_npf[high_impact, j, i]
                                )

            for j in range(cands_per_variable["vtx"]):
                for i in range(vars_per_candidate["vtx"]):
                    if i in integer_variables_by_candidate["vtx"]:
                        adv_vtx[:, j, i] = vtx[:, j, i]
                    else:
                        defaults_vtx = (
                            vtx[:, j, i].cpu() == defaults_per_variable["vtx"][i]
                        )
                        if torch.sum(defaults_vtx) != 0:
                            adv_vtx[:, j, i][defaults_vtx] = vtx[:, j, i][defaults_vtx]

                        if restrict_impact > 0:
                            difference = adv_vtx[:, j, i] - vtx[:, j, i]
                            allowed_perturbation = restrict_impact * torch.abs(
                                vtx[:, j, i]
                            )
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact) != 0:
                                adv_vtx[high_impact, j, i] = (
                                    vtx[high_impact, j, i]
                                    + allowed_perturbation[high_impact]
                                    * dx_vtx[high_impact, j, i]
                                )

    np.save(
        f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/glob_{batch_index}_-1.npy",
        adv_glob.detach().cpu().numpy(),
    )
    np.save(
        f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/cpf_{batch_index}_-1.npy",
        adv_cpf.detach().cpu().numpy(),
    )
    np.save(
        f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/npf_{batch_index}_-1.npy",
        adv_npf.detach().cpu().numpy(),
    )
    np.save(
        f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/vtx_{batch_index}_-1.npy",
        adv_vtx.detach().cpu().numpy(),
    )

    return adv_glob.detach(), adv_cpf.detach(), adv_npf.detach(), adv_vtx.detach()
"""


def pgd_attack(
    epsilon=1e-2,
    pgd_loops=-1,
    sample=None,
    targets=None,
    thismodel=None,
    thiscriterion=None,
    reduced=True,
    dev=torch.device("cuda"),
    restrict_impact=-1,
    epsilon_factors=True,
    batch_index=None,
):
    if epsilon == 0:
        return sample

    if epsilon_factors:
        epsilon_glob = torch.from_numpy(np.load(epsilons_per_feature["glob"])).to(dev)
        epsilon_cpf = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["cpf"]))
        ).to(dev)
        epsilon_npf = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["npf"]))
        ).to(dev)
        epsilon_vtx = torch.from_numpy(
            np.transpose(np.load(epsilons_per_feature["vtx"]))
        ).to(dev)
    else:
        epsilon_glob = 1.0
        epsilon_cpf = 1.0
        epsilon_npf = 1.0
        epsilon_vtx = 1.0

    alpha_glob = epsilon * epsilon_glob / pgd_loops
    alpha_cpf = epsilon * epsilon_cpf / pgd_loops
    alpha_npf = epsilon * epsilon_npf / pgd_loops
    alpha_vtx = epsilon * epsilon_vtx / pgd_loops

    glob, cpf, npf, vtx = sample

    adv_glob = glob.clone().detach().to(dev)
    adv_cpf = cpf.clone().detach().to(dev)
    adv_npf = npf.clone().detach().to(dev)
    adv_vtx = vtx.clone().detach().to(dev)

    adv_glob.requires_grad = True
    adv_cpf.requires_grad = True
    adv_npf.requires_grad = True
    adv_vtx.requires_grad = True

    dx_glob = 0
    dx_cpf = 0
    dx_npf = 0
    dx_vtx = 0

    for l in range(pgd_loops):
        prediction = thismodel(adv_glob, adv_cpf, adv_npf, adv_vtx)

        loss = thiscriterion(prediction, targets)

        thismodel.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_glob = adv_glob.grad.detach().sign()
            grad_cpf = adv_cpf.grad.detach().sign()
            grad_npf = adv_npf.grad.detach().sign()
            grad_vtx = adv_vtx.grad.detach().sign()

            delta_glob = torch.clamp(
                adv_glob - (adv_glob + alpha_glob * grad_glob),
                min=-epsilon * epsilon_glob,
                max=epsilon * epsilon_glob,
            )
            delta_cpf = torch.clamp(
                adv_cpf - (adv_cpf + alpha_cpf * grad_cpf),
                min=-epsilon * epsilon_cpf,
                max=epsilon * epsilon_cpf,
            )
            delta_npf = torch.clamp(
                adv_npf - (adv_npf + alpha_npf * grad_npf),
                min=-epsilon * epsilon_npf,
                max=epsilon * epsilon_npf,
            )
            delta_vtx = torch.clamp(
                adv_vtx - (adv_vtx + alpha_vtx * grad_vtx),
                min=-epsilon * epsilon_vtx,
                max=epsilon * epsilon_vtx,
            )

            adv_glob -= delta_glob
            adv_cpf -= delta_cpf
            adv_npf -= delta_npf
            adv_vtx -= delta_vtx

            dx_glob += grad_glob
            dx_cpf += grad_cpf
            dx_npf += grad_npf
            dx_vtx += grad_vtx
            """
            np.save(
                f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/glob_{batch_index}_{l}.npy",
                adv_glob.detach().cpu().numpy(),
            )
            np.save(
                f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/cpf_{batch_index}_{l}.npy",
                adv_cpf.detach().cpu().numpy(),
            )
            np.save(
                f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/npf_{batch_index}_{l}.npy",
                adv_npf.detach().cpu().numpy(),
            )
            np.save(
                f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/vtx_{batch_index}_{l}.npy",
                adv_vtx.detach().cpu().numpy(),
            )
            """
    with torch.no_grad():
        if reduced:
            for i in range(vars_per_candidate["glob"]):
                if (
                    i in integer_variables_by_candidate["glob"]
                ):  # don't change integer variables
                    adv_glob[:, i] = glob[:, i]
                else:  # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:, i].cpu() == defaults_per_variable["glob"][i]
                    if torch.sum(defaults_glob) != 0:  # don't change default varibles
                        adv_glob[:, i][defaults_glob] = glob[:, i][defaults_glob]

            for j in range(cands_per_variable["cpf"]):
                for i in range(vars_per_candidate["cpf"]):
                    if i in integer_variables_by_candidate["cpf"]:
                        adv_cpf[:, j, i] = cpf[:, j, i]
                    else:
                        defaults_cpf = (
                            cpf[:, j, i].cpu() == defaults_per_variable["cpf"][i]
                        )
                        if torch.sum(defaults_cpf) != 0:
                            adv_cpf[:, j, i][defaults_cpf] = cpf[:, j, i][defaults_cpf]

            for j in range(cands_per_variable["npf"]):
                for i in range(vars_per_candidate["npf"]):
                    if i in integer_variables_by_candidate["npf"]:
                        adv_npf[:, j, i] = npf[:, j, i]
                    else:
                        defaults_npf = (
                            npf[:, j, i].cpu() == defaults_per_variable["npf"][i]
                        )
                        if torch.sum(defaults_npf) != 0:
                            adv_npf[:, j, i][defaults_npf] = npf[:, j, i][defaults_npf]

            for j in range(cands_per_variable["vtx"]):
                for i in range(vars_per_candidate["vtx"]):
                    if i in integer_variables_by_candidate["vtx"]:
                        adv_vtx[:, j, i] = vtx[:, j, i]
                    else:
                        defaults_vtx = (
                            vtx[:, j, i].cpu() == defaults_per_variable["vtx"][i]
                        )
                        if torch.sum(defaults_vtx) != 0:
                            adv_vtx[:, j, i][defaults_vtx] = vtx[:, j, i][defaults_vtx]

        if restrict_impact > 0:
            adv_glob = torch.clamp(
                adv_glob,
                min=glob - restrict_impact * torch.abs(glob),
                max=glob + restrict_impact * torch.abs(glob),
            )
            adv_cpf = torch.clamp(
                adv_cpf,
                min=cpf - restrict_impact * torch.abs(cpf),
                max=cpf + restrict_impact * torch.abs(cpf),
            )
            adv_npf = torch.clamp(
                adv_npf,
                min=npf - restrict_impact * torch.abs(npf),
                max=npf + restrict_impact * torch.abs(npf),
            )
            adv_vtx = torch.clamp(
                adv_vtx,
                min=vtx - restrict_impact * torch.abs(vtx),
                max=vtx + restrict_impact * torch.abs(vtx),
            )
    """
    np.save(
        f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/glob_{batch_index}_-1.npy",
        adv_glob.detach().cpu().numpy(),
    )
    np.save(
        f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/cpf_{batch_index}_-1.npy",
        adv_cpf.detach().cpu().numpy(),
    )
    np.save(
        f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/npf_{batch_index}_-1.npy",
        adv_npf.detach().cpu().numpy(),
    )
    np.save(
        f"/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/per_loop_new/vtx_{batch_index}_-1.npy",
        adv_vtx.detach().cpu().numpy(),
    )
    """
    return adv_glob.detach(), adv_cpf.detach(), adv_npf.detach(), adv_vtx.detach()
