#!/usr/bin/env python3
from argparse import ArgumentParser

parser = ArgumentParser('Apply a model to a (test) source sample.')
parser.add_argument('model')
parser.add_argument('inputModel')
parser.add_argument('trainingDataCollection', help="the training data collection. Used to infer data format and batch size.")
parser.add_argument('inputSourceFileList', help="can be text file or a DataCollection file in the same directory as the sample files, or just a single traindata file.")
parser.add_argument('outputDir', help="will be created if it doesn't exist.")
parser.add_argument("-b", help="batch size, overrides the batch size from the training data collection.",default="-1")
parser.add_argument("--gpu",  help="select specific GPU", metavar="OPT", default="")
parser.add_argument("--unbuffered", help="do not read input in memory buffered mode (for lower memory consumption on fast disks)", default=False, action="store_true")
parser.add_argument("--pad_rowsplits", help="pad the row splits if the input is ragged", default=False, action="store_true")
parser.add_argument("-attack", help="use adversarial attack (Noise|FGSM|PGD) or leave blank to use undisturbed features only", default="")
parser.add_argument("-att_magnitude", help="distort input features with adversarial attack, using specified magnitude of attack", default="-1")
parser.add_argument("-restrict_impact", help="limit attack impact to this fraction of the input value (percent-cap on distortion)", default="-1")
parser.add_argument("-pgd_loops", help="define number of iterations in the PGD attack", default="-1")

args = parser.parse_args()
batchsize = int(args.b)
attack = args.attack
att_magnitude = float(args.att_magnitude)
restrict_impact = float(args.restrict_impact)
pgd_loops = int(args.pgd_loops)

import imp
import numpy as np
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
import tempfile
import atexit
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_deepjet import DeepJet
from pytorch_deepjet_run2 import DeepJet_Run2
from pytorch_deepjet_transformer import DeepJetTransformer
from torch.optim import Adam, SGD
from tqdm import tqdm
from attacks import apply_noise, fgsm_attack, pgd_attack
inputdatafiles=[]
inputdir=None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_dir = "/eos/user/a/aljung/DeepJet/Train_DF_Run2/adam/checkpoint_epoch_"
checkpoints = []
for i in range(101):
    checkpoints.append(f"{base_dir}{i+1}.pth")
        
base_dir = "/eos/user/a/aljung/DeepJet/Train_DF_Run2/adam/predictions/"
pred_dir = []
text_dir = []
for i in range(39):
    pred_dir.append(f"{base_dir}prediction_epoch_{i+1}.root")
    text_dir.append(f"{base_dir}outfiles{i+1}.txt")

from definitions import epsilons_per_feature, vars_per_candidate

glob_vars = vars_per_candidate['glob']

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

def test_loop(dataloader, model, nbatches, attack = "", att_magnitude = -1., restrict_impact = -1., pgd_loops = pgd_loops, loss_fn = cross_entropy_one_hot, epsilon_factors=None):
    predictions = 0
    
    for b, (features_list, truth_list) in enumerate(dataloader):
        glob = torch.Tensor(features_list[0]).to(device)
        cpf = torch.Tensor(features_list[1]).to(device)
        npf = torch.Tensor(features_list[2]).to(device)
        vtx = torch.Tensor(features_list[3]).to(device)
        y = torch.Tensor(truth_list[0]).to(device)    

        glob[:,:] = torch.where(glob[:,:] == -999., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])
        glob[:,:] = torch.where(glob[:,:] ==   -1., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])

        pred = nn.Softmax(dim=1)(model(glob,cpf,npf,vtx)).cpu().detach().numpy()
        if b == 0:
            predictions = pred
        else:
            predictions = np.concatenate((predictions, pred), axis=0)
        desc = 'Predicting probs : '
    
    return predictions

## prepare input lists for different file formats
if args.inputSourceFileList[-6:] == ".djcdc":
    print('reading from data collection',args.inputSourceFileList)
    predsamples = DataCollection(args.inputSourceFileList)
    inputdir = predsamples.dataDir
    for s in predsamples.samples:
        inputdatafiles.append(s)
        
elif args.inputSourceFileList[-6:] == ".djctd":
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    infile = os.path.basename(args.inputSourceFileList)
    inputdatafiles.append(infile)
else:
    print('reading from text file',args.inputSourceFileList)
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    with open(args.inputSourceFileList, "r") as f:
        for s in f:
            inputdatafiles.append(s.replace('\n', '').replace(" ",""))

if torch.cuda.is_available():
    torch.backends.cudnn.enabled=False

if args.model == 'DeepJet':
    model = DeepJet(num_classes = 6)
if args.model == 'DeepJet_Run2':
    model = DeepJet_Run2(num_classes = 6)
if args.model == 'DeepJetTransformer':
    model = DeepJetTransformer(num_classes = 4)

dc = None
if args.inputSourceFileList[-6:] == ".djcdc" and not args.trainingDataCollection[-6:] == ".djcdc":
    dc = DataCollection(args.inputSourceFileList)
    if batchsize < 1:
        batchsize = 1
    print('No training data collection given. Using batch size of',batchsize)
else:
    dc = DataCollection(args.trainingDataCollection)

outputs = []
os.system('mkdir -p '+args.outputDir)

for inputfile in inputdatafiles:
    
    print('predicting ',inputdir+"/"+inputfile)
    
    use_inputdir = inputdir
    if inputfile[0] == "/":
        use_inputdir=""
    else:
        use_inputdir=use_inputdir+"/"
    outfilename = "pred_"+os.path.basename( inputfile ) #root_tuple_merged.root
    
    td = dc.dataclass()

    if inputfile[-5:] == 'djctd':
        if args.unbuffered:
            td.readFromFile(use_inputdir+inputfile)
        else:
            td.readFromFileBuffered(use_inputdir+inputfile)
    else:
        print('converting '+inputfile)
        print(use_inputdir+inputfile)
        td.readFromSourceFile(use_inputdir+inputfile, dc.weighterobjects, istraining=False)

    epsilon_factors = {'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(device), 'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(device), 'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(device), 'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(device),}
    
    for i in range(len(checkpoints)):
        print(f"Predicting on {checkpoints[i]}")
        check = torch.load(checkpoints[i], map_location=device)
        model.load_state_dict(check['state_dict'])
        model.to(device)
        model.eval()
        
        td.readFromSourceFile(use_inputdir+inputfile, dc.weighterobjects, istraining=False)
        gen = TrainDataGenerator()
        if batchsize < 1:
            batchsize = dc.getBatchSize()
        gen.setBatchSize(batchsize)
        gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
        gen.setSkipTooLargeBatches(False)
        gen.setBuffer(td) 
        test_generator = gen.feedNumpyData()
    
        predicted = test_loop(test_generator, model, nbatches=gen.getNBatches(), attack = attack, att_magnitude = att_magnitude, restrict_impact = restrict_impact, epsilon_factors=epsilon_factors)
        
        x = td.transferFeatureListToNumpy(args.pad_rowsplits)
        w = td.transferWeightListToNumpy(args.pad_rowsplits)
        y = td.transferTruthListToNumpy(args.pad_rowsplits)
    
        if not type(predicted) == list:
            predicted = [predicted]   
        
        overwrite_outname = td.writeOutPrediction(predicted, x, y, w, pred_dir[i], use_inputdir+"/"+inputfile)
        if overwrite_outname is not None:
            outfilename = overwrite_outname
        outputs.append(outfilename)
    
        with open(text_dir[i],"w") as f:
            for l in outputs:
                f.write(l+'\n')
                
        gen.clear()
        td.clear()
        
        print(f"Finished predicting on {checkpoints[i]}")
