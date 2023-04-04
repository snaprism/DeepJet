import os
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ROOT
import numpy.lib.recfunctions as rf
import uproot as u

base_dir = "/eos/user/a/aljung/DeepJet/Train_DF_Run2/adam/"
pred_dir = []
roc_dir = []
for i in range(39):
    pred_dir.append(f"{base_dir}predictions/prediction_epoch_{i+1}.npy")
    roc_dir.append(f"{i+1}")


def spit_out_roc(disc,truth_array,selection_array):
    tprs = pd.DataFrame()
    truth = truth_array[selection_array]*1
    disc = disc[selection_array]
    tmp_fpr, tmp_tpr, _ = roc_curve(truth, disc)
    coords = pd.DataFrame()
    coords['fpr'] = tmp_fpr
    coords['tpr'] = tmp_tpr
    clean = coords.drop_duplicates(subset=['fpr'])
    auc_ = auc(clean.fpr,clean.tpr)
    print('AUC: ', str(auc_))
    return clean.tpr, clean.fpr, auc_


pred = []
isDeepJet = True
if isDeepJet:
    listbranch = ['prob_isB', 'prob_isBB','prob_isLeptB', 'prob_isC','prob_isUDS','prob_isG','isB', 'isBB', 'isLeptB', 'isC','isUDS','isG','jet_pt', 'jet_eta']
else:
    listbranch = ['prob_isB', 'prob_isBB', 'prob_isC','prob_isUDSG','isB', 'isBB', 'isC','isUDSG','jet_pt', 'jet_eta']

for i in range(len(pred_dir)):
    print(f"processing {i}/{len(pred_dir)}")
    nparray = rf.structured_to_unstructured(np.array(np.load(pred_dir[i])))

    df = np.core.records.fromarrays([nparray[:,k] for k in range(len(listbranch))],names=listbranch)

    if isDeepJet:
        b_jets = df['isB']+df['isBB']+df['isLeptB']
        c_jets = df['isC']
        b_out = df['prob_isB']+df['prob_isBB']+df['prob_isLeptB']
        c_out = df['prob_isC']
        light_out = df['prob_isUDS']+df['prob_isG']
        bvsl = np.where((b_out + light_out)!=0,
                        (b_out)/(b_out + light_out),
                        -1)
        cvsb = np.where((b_out + c_out)!=0,
                        (c_out)/(b_out + c_out),
                        -1)
        cvsl = np.where((light_out + c_out)!=0,
                        (c_out)/(light_out + c_out),
                        -1)
        summed_truth = df['isB']+df['isBB']+df['isLeptB']+df['isC']+df['isUDS']+df['isG']

        veto_b = (df['isB'] != 1) & (df['isBB'] != 1) & (df['isLeptB'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)
        veto_c = (df['isC'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)
        veto_udsg = (df['isUDS'] != 1) & (df['isG'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)

    else:
        b_jets = df['isB']+df['isBB']
        b_disc = df['prob_isB']+df['prob_isBB']
        summed_truth = df['isB']+df['isBB']+df['isC']+df['isUDSG']
        veto_c = (df['isC'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)
        veto_udsg = (df['isUDSG'] != 1) & ( df['jet_pt'] > 30) & (summed_truth != 0)

    x1, y1, auc1 = spit_out_roc(bvsl,b_jets,veto_c)
    x2, y2, auc2 = spit_out_roc(cvsb,c_jets,veto_udsg)
    x3, y3, auc3 = spit_out_roc(cvsl,c_jets,veto_b)
    np.save(base_dir + 'roc/' + 'BvL_' + roc_dir[i] + '.npy', np.array([x1,y1,auc1],dtype=object))
    np.save(base_dir + 'roc/' + 'CvB_' + roc_dir[i] + '.npy', np.array([x2,y2,auc2],dtype=object))
    np.save(base_dir + 'roc/' + 'CvL_' + roc_dir[i] + '.npy', np.array([x3,y3,auc3],dtype=object))
