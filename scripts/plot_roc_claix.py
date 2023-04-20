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
import numpy.lib.recfunctions as rf

predictions = ["pgd_loops/1/predict/", "pgd_loops/1/predict_pgd_1/", "pgd_loops/2/predict/", "pgd_loops/2/predict_pgd_2/", "pgd_loops/3/predict/", "pgd_loops/3/predict_pgd_3/", "pgd_loops/4/predict/", "pgd_loops/4/predict_pgd_4/", "pgd_loops/5/predict/", "pgd_loops/5/predict_pgd_5/"]

def save_roc(prediction_path):
    base_dir         = "/hpcwork/pj214607/work/promotion/deepjet/results/"
    output_dirs      = [base_dir + f"{i}" for i in prediction_path]
    
    listbranch = ["prob_isB", "prob_isBB", "prob_isLeptB", "prob_isC", "prob_isUDS", "prob_isG", "isB", "isBB", "isLeptB", "isC", "isUDS", "isG", "jet_pt", "jet_eta"]

    def spit_out_roc(disc,truth_array,selection_array):
        tprs                = pd.DataFrame()
        truth               = truth_array[selection_array] * 1
        disc                = disc[selection_array]
        tmp_fpr, tmp_tpr, _ = roc_curve(truth, disc)
        coords              = pd.DataFrame()
        coords["fpr"]       = tmp_fpr
        coords["tpr"]       = tmp_tpr
        clean               = coords.drop_duplicates(subset=["fpr"])
        auc_                = auc(clean.fpr, clean.tpr)
        print("AUC: ", str(auc_))
        print("\n")
        return clean.tpr, clean.fpr, auc_
    
    for j,output in enumerate(output_dirs):
        nparray  = rf.structured_to_unstructured(np.array(np.load(output + "pred_ntuple_merged_342.npy")))
        df       = np.core.records.fromarrays([nparray[:,k] for k in range(len(listbranch))], names=listbranch)

        b_jets    = df["isB"] + df["isBB"] + df["isLeptB"]
        c_jets    = df["isC"]
        b_out     = df["prob_isB"] + df["prob_isBB"] + df["prob_isLeptB"]
        c_out     = df["prob_isC"]
        light_out = df["prob_isUDS"] + df["prob_isG"]
        
        bvsl = np.where((b_out + light_out)!=0, (b_out)/(b_out + light_out), -1)
        cvsb = np.where((b_out + c_out)!=0, (c_out)/(b_out + c_out), -1)
        cvsl = np.where((light_out + c_out)!=0, (c_out)/(light_out + c_out), -1)
        
        summed_truth = df["isB"] + df["isBB"] + df["isLeptB"] + df["isC"] + df["isUDS"] + df["isG"]

        veto_b    = (df["isB"] != 1) & (df["isBB"] != 1) & (df["isLeptB"] != 1) & ( df["jet_pt"] > 30) & (summed_truth != 0)
        veto_c    = (df["isC"] != 1) & ( df["jet_pt"] > 30) & (summed_truth != 0)
        veto_udsg = (df["isUDS"] != 1) & (df["isG"] != 1) & ( df["jet_pt"] > 30) & (summed_truth != 0)

        x1, y1, auc1 = spit_out_roc(bvsl,b_jets,veto_c)
        x2, y2, auc2 = spit_out_roc(cvsb,c_jets,veto_udsg)
        x3, y3, auc3 = spit_out_roc(cvsl,c_jets,veto_b)
        
        np.save(output + "BvL.npy", np.array([x1,y1,auc1], dtype=object))
        np.save(output + "CvB.npy", np.array([x2,y2,auc2], dtype=object))
        np.save(output + "CvL.npy", np.array([x3,y3,auc3], dtype=object))
        
save_roc(predictions)