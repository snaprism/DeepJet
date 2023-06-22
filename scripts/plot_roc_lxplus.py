print("start import")
import os
import matplotlib

matplotlib.use("Agg")
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ROOT
import uproot as u

print("finish import")

model_name = "nominal_seed_0"
prediction_setup = ""
prediction_files = "outfiles"


def spit_out_roc(disc, truth_array, selection_array):
    tprs = pd.DataFrame()
    truth = truth_array[selection_array] * 1
    disc = disc[selection_array]
    tmp_fpr, tmp_tpr, _ = roc_curve(truth, disc)
    coords = pd.DataFrame()
    coords["fpr"] = tmp_fpr
    coords["tpr"] = tmp_tpr
    clean = coords.drop_duplicates(subset=["fpr"])
    auc_ = auc(clean.fpr, clean.tpr)
    print("AUC: ", str(auc_))
    return clean.tpr, clean.fpr, auc_


pred = []
isDeepJet = True
if isDeepJet:
    listbranch = [
        "prob_isB",
        "prob_isBB",
        "prob_isLeptB",
        "prob_isC",
        "prob_isUDS",
        "prob_isG",
        "isB",
        "isBB",
        "isLeptB",
        "isC",
        "isUDS",
        "isG",
        "jet_pt",
        "jet_eta",
    ]
else:
    listbranch = [
        "prob_isB",
        "prob_isBB",
        "prob_isC",
        "prob_isUDSG",
        "isB",
        "isBB",
        "isC",
        "isUDSG",
        "jet_pt",
        "jet_eta",
    ]

dirz = f"/eos/user/a/aljung/DeepJet/Train_DF_Run2/test/"
truthfile = open(dirz + prediction_files + ".txt", "r")

config_name = model_name + prediction_setup + "_" + prediction_files

print("opened text file")
count = 0
import numpy.lib.recfunctions as rf

for i, line in enumerate(truthfile):
    count += 1
    if len(line) < 1:
        continue
    file1name = str(dirz + line.split("\n")[0])
    events = rf.structured_to_unstructured(
        np.array(np.load(file1name.strip(".root") + ".npy"))
    )
    nparray = events if i == 0 else np.concatenate((nparray, events))

print("added files")

print(type(nparray))
print(len(nparray))
print(type(nparray[0]))

df = np.core.records.fromarrays(
    [nparray[:, k] for k in range(len(listbranch))], names=listbranch
)
print("converted to df")

if isDeepJet:
    b_jets = df["isB"] + df["isBB"] + df["isLeptB"]
    c_jets = df["isC"]
    b_out = df["prob_isB"] + df["prob_isBB"] + df["prob_isLeptB"]
    c_out = df["prob_isC"]
    light_out = df["prob_isUDS"] + df["prob_isG"]
    bvsl = np.where((b_out + light_out) != 0, (b_out) / (b_out + light_out), -1)
    cvsb = np.where((b_out + c_out) != 0, (c_out) / (b_out + c_out), -1)
    cvsl = np.where((light_out + c_out) != 0, (c_out) / (light_out + c_out), -1)
    summed_truth = (
        df["isB"] + df["isBB"] + df["isLeptB"] + df["isC"] + df["isUDS"] + df["isG"]
    )

    veto_b = (
        (df["isB"] != 1)
        & (df["isBB"] != 1)
        & (df["isLeptB"] != 1)
        & (df["jet_pt"] > 30)
        & (summed_truth != 0)
    )
    veto_c = (df["isC"] != 1) & (df["jet_pt"] > 30) & (summed_truth != 0)
    veto_udsg = (
        (df["isUDS"] != 1)
        & (df["isG"] != 1)
        & (df["jet_pt"] > 30)
        & (summed_truth != 0)
    )

else:
    b_jets = df["isB"] + df["isBB"]
    b_disc = df["prob_isB"] + df["prob_isBB"]
    summed_truth = df["isB"] + df["isBB"] + df["isC"] + df["isUDSG"]
    veto_c = (df["isC"] != 1) & (df["jet_pt"] > 30) & (summed_truth != 0)
    veto_udsg = (df["isUDSG"] != 1) & (df["jet_pt"] > 30) & (summed_truth != 0)

x1, y1, auc1 = spit_out_roc(bvsl, b_jets, veto_c)
x2, y2, auc2 = spit_out_roc(cvsb, c_jets, veto_udsg)
x3, y3, auc3 = spit_out_roc(cvsl, c_jets, veto_b)
np.save(dirz + f"BvL_{prediction_files}.npy", np.array([x1, y1, auc1], dtype=object))
np.save(dirz + f"CvB_{prediction_files}.npy", np.array([x2, y2, auc2], dtype=object))
np.save(dirz + f"CvL_{prediction_files}.npy", np.array([x3, y3, auc3], dtype=object))
