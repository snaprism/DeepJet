#!/usr/bin/env python3
from definitions import epsilons_per_feature, vars_per_candidate
from pytorch_deepjet_transformer import DeepJetTransformer
from attacks import apply_noise, fgsm_attack, pgd_attack
from DeepJetCore.dataPipeline import TrainDataGenerator
from DeepJetCore.DataCollection import DataCollection
from pytorch_deepjet_run2 import DeepJet_Run2
from argparse import ArgumentParser
from pytorch_deepjet import DeepJet
from torch.optim import Adam, SGD
import torch.nn.functional as F
from IPython import embed
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import tempfile
import atexit
import torch
import sys
import imp
import os


parser = ArgumentParser("Apply a model to a (test) source sample.")
parser.add_argument("model")
parser.add_argument("inputModel")
parser.add_argument(
    "trainingDataCollection",
    help="the training data collection. Used to infer data format and batch size.",
)
parser.add_argument(
    "inputSourceFileList",
    help="can be text file or a DataCollection file in the same directory as the sample files, or just a single traindata file.",
)
parser.add_argument("outputDir", help="will be created if it doesn't exist.")
parser.add_argument(
    "-b",
    help="batch size, overrides the batch size from the training data collection.",
    default="-1",
)
parser.add_argument("--gpu", help="select specific GPU", metavar="OPT", default="")
parser.add_argument(
    "--unbuffered",
    help="do not read input in memory buffered mode (for lower memory consumption on fast disks)",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--pad_rowsplits",
    help="pad the row splits if the input is ragged",
    default=False,
    action="store_true",
)
parser.add_argument(
    "-attack",
    help="use adversarial attack (Noise|FGSM|PGD) or leave blank to use undisturbed features only",
    default="",
)
parser.add_argument(
    "-att_magnitude",
    help="distort input features with adversarial attack, using specified magnitude of attack",
    default="-1",
)
parser.add_argument(
    "-restrict_impact",
    help="limit attack impact to this fraction of the input value (percent-cap on distortion)",
    default="-1",
)
parser.add_argument(
    "-pgd_loops", help="define number of iterations in the PGD attack", default="-1"
)

parser.add_argument(
    "-jetfool_loops", help="define number of iterations in the jetfool attack", default="5"
)

args = parser.parse_args()
batchsize = int(args.b)
attack = args.attack
att_magnitude = float(args.att_magnitude)
restrict_impact = float(args.restrict_impact)
pgd_loops = int(args.pgd_loops)
jetfool_loops = int(args.jetfool_loops)

inputdatafiles = []
inputdir = None

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print(f"This process has the PID {os.getpid()} .")

glob_vars = vars_per_candidate["glob"]


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def test_loop(
    dataloader,
    model,
    nbatches,
    pbar,
    attack="",
    att_magnitude=-1.0,
    restrict_impact=-1.0,
    pgd_loops=pgd_loops,
    loss_fn=cross_entropy_one_hot,
    epsilon_factors=None,
):
    predictions = 0

    glob_adv_liste = []
    cpf_adv_liste = []
    npf_adv_liste = []
    vtx_adv_liste = []
    y_liste = []
    # with torch.no_grad():
    for b in range(nbatches):
        features_list, truth_list = next(dataloader)
        glob = torch.Tensor(features_list[0]).to(device)
        cpf = torch.Tensor(features_list[1]).to(device)
        npf = torch.Tensor(features_list[2]).to(device)
        vtx = torch.Tensor(features_list[3]).to(device)
        y = torch.Tensor(truth_list[0]).to(device)

        glob[:, :] = torch.where(
            glob[:, :] == -999.0,
            torch.zeros(len(glob), glob_vars).to(device),
            glob[:, :],
        )
        glob[:, :] = torch.where(
            glob[:, :] == -1.0, torch.zeros(len(glob), glob_vars).to(device), glob[:, :]
        )

        if attack == "Noise":
            glob = apply_noise(
                glob,
                magn=att_magnitude,
                offset=[0],
                dev=device,
                restrict_impact=restrict_impact,
                var_group="glob",
            )
            cpf = apply_noise(
                cpf,
                magn=att_magnitude,
                offset=[0],
                dev=device,
                restrict_impact=restrict_impact,
                var_group="cpf",
            )
            npf = apply_noise(
                npf,
                magn=att_magnitude,
                offset=[0],
                dev=device,
                restrict_impact=restrict_impact,
                var_group="npf",
            )
            vtx = apply_noise(
                vtx,
                magn=att_magnitude,
                offset=[0],
                dev=device,
                restrict_impact=restrict_impact,
                var_group="vtx",
            )

        elif attack == "FGSM":
            glob, cpf, npf, vtx = fgsm_attack(
                sample=(glob, cpf, npf, vtx),
                epsilon=att_magnitude,
                dev=device,
                targets=y,
                thismodel=model,
                thiscriterion=loss_fn,
                restrict_impact=restrict_impact,
                epsilon_factors=epsilon_factors,
            )

        elif attack == "PGD":
            glob, cpf, npf, vtx = pgd_attack(
                sample=(glob, cpf, npf, vtx),
                epsilon=att_magnitude,
                pgd_loops=pgd_loops,
                dev=device,
                targets=y,
                thismodel=model,
                thiscriterion=loss_fn,
                restrict_impact=restrict_impact,
                epsilon_factors=epsilon_factors,
                batch_index=b,
            )

        elif attack == "jetfool":
            sys.path.append("/home/home1/institut_3a/ajung/work/repositories/jetfool")
            from jetfool import jetfool
            glob, cpf, npf, vtx = jetfool((glob, cpf, npf, vtx), model, batchsize, device, jetfool_loops, restrict_impact, number_classes=6, overshoot=0.02)

        glob_adv_liste.append(glob.detach().cpu().numpy())
        cpf_adv_liste.append(cpf.detach().cpu().numpy())
        npf_adv_liste.append(npf.detach().cpu().numpy())
        vtx_adv_liste.append(vtx.detach().cpu().numpy())
        y_liste.append(y.detach().cpu().numpy())
        
        pred = nn.Softmax(dim=1)(model(glob, cpf, npf, vtx)).cpu().detach().numpy()
        if b == 0:
            predictions = pred
        else:
            predictions = np.concatenate((predictions, pred), axis=0)
        desc = "Predicting probs : "
        pbar.set_description(desc)
        pbar.update(1)

    np.save(
        "/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/nominal_jetfool/glob_batch.npy",
        np.asarray(glob_adv_liste),
    )
    np.save(
        "/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/nominal_jetfool/cpf_batch.npy",
        np.asarray(cpf_adv_liste),
    )
    np.save(
        "/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/nominal_jetfool/npf_batch.npy",
        np.asarray(npf_adv_liste),
    )
    np.save(
        "/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/nominal_jetfool/vtx_batch.npy",
        np.asarray(vtx_adv_liste),
    )
    np.save(
        "/net/scratch_cms3a/ajung/deepjet/data/one_sample/numpy/nominal_jetfool/y_batch.npy",
        np.asarray(y_liste),
    )

    return predictions


# prepare input lists for different file formats
if args.inputSourceFileList[-6:] == ".djcdc":
    print("reading from data collection", args.inputSourceFileList)
    predsamples = DataCollection(args.inputSourceFileList)
    inputdir = predsamples.dataDir
    for s in predsamples.samples:
        inputdatafiles.append(s)

elif args.inputSourceFileList[-6:] == ".djctd":
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    infile = os.path.basename(args.inputSourceFileList)
    inputdatafiles.append(infile)
else:
    print("reading from text file", args.inputSourceFileList)
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    with open(args.inputSourceFileList, "r") as f:
        for s in f:
            inputdatafiles.append(s.replace("\n", "").replace(" ", ""))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# https://github.com/pytorch/captum/issues/564#issuecomment-748274352
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

if args.model == "DeepJet":
    model = DeepJet(num_classes=6)
if args.model == "DeepJet_Run2":
    model = DeepJet_Run2(num_classes=6)
if args.model == "DeepJetTransformer":
    model = DeepJetTransformer(num_classes=4)

check = torch.load(args.inputModel, map_location=torch.device("cpu"))
model.load_state_dict(check["state_dict"])

model.to(device)
model.eval()

dc = None
if (
    args.inputSourceFileList[-6:] == ".djcdc"
    and not args.trainingDataCollection[-6:] == ".djcdc"
):
    dc = DataCollection(args.inputSourceFileList)
    if batchsize < 1:
        batchsize = 1
    print("No training data collection given. Using batch size of", batchsize)
else:
    dc = DataCollection(args.trainingDataCollection)

outputs = []
os.system("mkdir -p " + args.outputDir)

for inputfile in inputdatafiles:
    print("predicting ", inputdir + "/" + inputfile)

    use_inputdir = inputdir
    if inputfile[0] == "/":
        use_inputdir = ""
    else:
        use_inputdir = use_inputdir + "/"
    outfilename = "pred_" + os.path.basename(inputfile)

    td = dc.dataclass()

    if inputfile[-5:] == "djctd":
        if args.unbuffered:
            td.readFromFile(use_inputdir + inputfile)
        else:
            td.readFromFileBuffered(use_inputdir + inputfile)
    else:
        print("converting " + inputfile)
        print(use_inputdir + inputfile)
        td.readFromSourceFile(
            use_inputdir + inputfile, dc.weighterobjects, istraining=False
        )

    gen = TrainDataGenerator()
    if batchsize < 1:
        batchsize = dc.getBatchSize()
    print("batch size", batchsize)
    gen.setBatchSize(batchsize)
    gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)

    with tqdm(total=gen.getNBatches()) as pbar:
        pbar.set_description("Predicting : ")

    epsilon_factors = {
        "glob": torch.Tensor(np.load(epsilons_per_feature["glob"]).transpose()).to(
            device
        ),
        "cpf": torch.Tensor(np.load(epsilons_per_feature["cpf"]).transpose()).to(
            device
        ),
        "npf": torch.Tensor(np.load(epsilons_per_feature["npf"]).transpose()).to(
            device
        ),
        "vtx": torch.Tensor(np.load(epsilons_per_feature["vtx"]).transpose()).to(
            device
        ),
    }

    predicted = test_loop(
        gen.feedNumpyData(),
        model,
        nbatches=gen.getNBatches(),
        pbar=pbar,
        attack=attack,
        att_magnitude=att_magnitude,
        restrict_impact=restrict_impact,
        epsilon_factors=epsilon_factors,
    )

    x = td.transferFeatureListToNumpy(args.pad_rowsplits)
    w = td.transferWeightListToNumpy(args.pad_rowsplits)
    y = td.transferTruthListToNumpy(args.pad_rowsplits)

    td.clear()
    gen.clear()

    if (
        not type(predicted) == list
    ):  # circumvent that keras return only an array if there is just one list item
        predicted = [predicted]

    # Optimal would be to include the discriminators here
    overwrite_outname = td.writeOutPrediction(
        predicted,
        x,
        y,
        w,
        args.outputDir + "/" + outfilename,
        use_inputdir + "/" + inputfile,
    )
    if overwrite_outname is not None:
        outfilename = overwrite_outname
    outputs.append(outfilename)

with open(args.outputDir + "/outfiles.txt", "w") as f:
    for l in outputs:
        f.write(l + "\n")

print(f"Finished process {os.getpid()} .")
