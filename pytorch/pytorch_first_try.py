# Alter line 439f to force continue training

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from definitions import epsilons_per_feature, vars_per_candidate
from DeepJetCore.DataCollection import DataCollection
from argparse import ArgumentParser
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
from torch.optim import Adam, SGD
import torch.nn.functional as F
from pdb import set_trace
from tqdm import tqdm
import torch.nn as nn
from attacks import *
import numpy as np
import shutil
import torch
import copy
import imp
import sys
import os

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

glob_vars = vars_per_candidate["glob"]


def train_loop(
    dataloader,
    nbatches,
    model,
    loss_fn,
    optimizer,
    device,
    epoch,
    epoch_pbar,
    attack,
    att_magnitude,
    restrict_impact,
    epsilon_factors,
    pgd_loops,
    acc_loss,
):
    for b in range(nbatches):
        # should not happen unless files are broken (will give additional errors)
        # if dataloader.isEmpty():
        #   raise Exception("ran out of data")

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
            )

        pred = model(glob, cpf, npf, vtx)
        loss = loss_fn(pred, y.type_as(pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_loss += loss.item()

        avg_loss = acc_loss / (b + 1)
        desc = f"Epoch {epoch+1} - loss {avg_loss:.6f}"
        epoch_pbar.set_description(desc)
        epoch_pbar.update(1)

    return avg_loss


def train_loop_(
    scheduler,
    dataloader,
    nbatches,
    model,
    loss_fn,
    optimizer,
    device,
    epoch,
    epoch_pbar,
    attack,
    att_magnitude,
    restrict_impact,
    epsilon_factors,
    pgd_loops,
    valgen,
    nbatches_val,
    model_,
    loss_fn_,
    device_,
    epoch_,
    acc_loss,
):
    for b in range(nbatches):
        # should not happen unless files are broken (will give additional errors)
        # if dataloader.isEmpty():
        #   raise Exception("ran out of data")

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
            )

        pred = model(glob, cpf, npf, vtx)
        loss = loss_fn(pred, y.type_as(pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_loss += loss.item()

        avg_loss = acc_loss / (b + 1)
        desc = f"Epoch {epoch+1} - loss {avg_loss:.6f}"
        epoch_pbar.set_description(desc)
        epoch_pbar.update(1)

        if b == 101:
            break

        model.eval()
        valgen.prepareNextEpoch()
        nbatches_val = valgen.getNBatches()
        val_generator = valgen.feedNumpyData()

        val_loss = val_loop_(val_generator, nbatches_val, model, loss_fn, device, epoch)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scheduler": scheduler.state_dict(),
            "best_loss": None,
            "train_loss": avg_loss,
            "val_loss": val_loss,
        }

        torch.save(
            checkpoint,
            "/eos/user/a/aljung/DeepJet/Train_DF_Run2/test/more_batch_checkpoints/checkpoint_epoch_"
            + str(epoch)
            + "_batch_"
            + str(b + 1)
            + ".pth",
        )
        model.train()
    return avg_loss


def val_loop(dataloader, nbatches, model, loss_fn, device, epoch):
    num_batches = nbatches
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for b in range(nbatches):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # should not happen unless files are broken (will give additional errors)
            # if dataloader.isEmpty():
            #   raise Exception("ran out of data")

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
                glob[:, :] == -1.0,
                torch.zeros(len(glob), glob_vars).to(device),
                glob[:, :],
            )

            _, labels = y.max(dim=1)
            pred = model(glob, cpf, npf, vtx)

            total += cpf.shape[0]
            test_loss += loss_fn(pred, y.type_as(pred)).item()
            avg_loss = test_loss / (b + 1)
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    correct /= total
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.6f}%, Avg loss: {avg_loss:>6f} \n"
    )
    return avg_loss


def val_loop_(dataloader, nbatches, model, loss_fn, device, epoch):
    num_batches = nbatches
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for b, (features_list, truth_list) in enumerate(dataloader):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # should not happen unless files are broken (will give additional errors)
            # if dataloader.isEmpty():
            #   raise Exception("ran out of data")
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
                glob[:, :] == -1.0,
                torch.zeros(len(glob), glob_vars).to(device),
                glob[:, :],
            )

            _, labels = y.max(dim=1)
            pred = model(glob, cpf, npf, vtx)
            total += cpf.shape[0]
            test_loss += loss_fn(pred, y.type_as(pred)).item()
            avg_loss = test_loss / (b + 1)
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    correct /= total
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.6f}%, Avg loss: {avg_loss:>6f} \n"
    )
    return avg_loss


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


class training_base(object):
    def __init__(
        self,
        model=None,
        criterion=cross_entropy_one_hot,
        optimizer=None,
        scheduler=None,
        splittrainandtest=0.85,
        useweights=False,
        testrun=False,
        testrun_fraction=0.1,
        resumeSilently=False,
        renewtokens=True,
        collection_class=DataCollection,
        parser=None,
        recreate_silently=False,
    ):
        import sys

        scriptname = sys.argv[0]

        parser = ArgumentParser("Run the training")
        parser.add_argument("inputDataCollection")
        parser.add_argument("outputDir")
        parser.add_argument(
            "--submitbatch",
            help="submits the job to condor",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--walltime",
            help="sets the wall time for the batch job, format: 1d5h or 2d or 3h etc",
            default="1d",
        )
        parser.add_argument(
            "--isbatchrun", help="is batch run", default=False, action="store_true"
        )

        args = parser.parse_args()

        self.inputData = os.path.abspath(args.inputDataCollection)
        self.outputDir = args.outputDir
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainedepoches = 0
        self.best_loss = np.inf
        self.checkpoint = 0

        isNewTraining = True
        if os.path.isdir(self.outputDir):
            if not (resumeSilently or recreate_silently):
                var = input('output dir exists. To recover a training, please type "yes"\n')
                # var = "yes"
                if not var == "yes":
                    raise Exception("output directory must not exist yet")
                isNewTraining = False
                if recreate_silently:
                    isNewTraining = True
        else:
            os.mkdir(self.outputDir)
        self.outputDir = os.path.abspath(self.outputDir)
        self.outputDir += "/"

        if recreate_silently:
            os.system("rm -rf " + self.outputDir + "*")

        # copy configuration to output dir
        if not args.isbatchrun:
            try:
                shutil.copyfile(
                    scriptname, self.outputDir + os.path.basename(scriptname)
                )
            except shutil.SameFileError:
                pass
            except BaseException as e:
                raise e

            self.copied_script = self.outputDir + os.path.basename(scriptname)
        else:
            self.copied_script = scriptname

        self.train_data = DataCollection()
        self.train_data.readFromFile(self.inputData)
        self.train_data.useweights = useweights

        self.val_data = self.train_data.split(splittrainandtest)

        if not isNewTraining:
            if os.path.isfile(self.outputDir + "/checkpoint.pth"):
                kfile = self.outputDir + "/checkpoint.pth"
            if os.path.isfile(kfile):
                print(kfile)

                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

                self.checkpoint = torch.load(kfile)
                self.trainedepoches = self.checkpoint["epoch"]
                self.best_loss = self.checkpoint["best_loss"]

                self.model.load_state_dict(self.checkpoint["state_dict"])
                self.model.to(self.device)
                self.optimizer.load_state_dict(self.checkpoint["optimizer"])
                # self.optimizer.to(self.device)
                self.scheduler.load_state_dict(self.checkpoint["scheduler"])

            else:
                print(
                    "no model found in existing output dir, starting training from scratch"
                )

    def saveModel(
        self,
        model,
        optimizer,
        epoch,
        scheduler,
        best_loss,
        train_loss,
        val_loss,
        is_best=False,
    ):
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        if is_best:
            torch.save(checkpoint, self.outputDir + "checkpoint_best_loss.pth")
        else:
            torch.save(checkpoint, self.outputDir + "checkpoint.pth")
        torch.save(
            checkpoint, self.outputDir + "checkpoint_epoch_" + str(epoch) + ".pth"
        )

    def _initTraining(self, batchsize, use_sum_of_squares=False):
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        self.train_data.batch_uses_sum_of_squares = use_sum_of_squares
        self.val_data.batch_uses_sum_of_squares = use_sum_of_squares

        self.train_data.writeToFile(self.outputDir + "trainsamples.djcdc")
        self.val_data.writeToFile(self.outputDir + "valsamples.djcdc")

        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)

    def trainModel(
        self,
        nepochs,
        batchsize,
        batchsize_use_sum_of_squares=False,
        extend_truth_list_by=0,
        load_in_mem=False,
        max_files=-1,
        plot_batch_loss=False,
        attack=None,
        att_magnitude=0.0,
        restrict_impact=-1,
        pgd_loops=-1,
        **trainargs,
    ):
        self._initTraining(batchsize, batchsize_use_sum_of_squares)
        print("starting training")
        if load_in_mem:
            print("make features")
            X_train = self.train_data.getAllFeatures(nfiles=max_files)
            X_test = self.val_data.getAllFeatures(nfiles=max_files)
            print("make truth")
            Y_train = self.train_data.getAllLabels(nfiles=max_files)
            Y_test = self.val_data.getAllLabels(nfiles=max_files)
            self.keras_model.fit(
                X_train,
                Y_train,
                batch_size=batchsize,
                epochs=nepochs,
                callbacks=self.callbacks.callbacks,
                validation_data=(X_test, Y_test),
                max_queue_size=1,
                use_multiprocessing=False,
                workers=0,
                **trainargs,
            )

        else:
            print("setting up generator... can take a while")
            traingen = self.train_data.invokeGenerator()
            valgen = self.val_data.invokeGenerator()
            traingen.setBatchSize(batchsize)
            valgen.setBatchSize(batchsize)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            epsilon_factors = {
                "glob": torch.Tensor(
                    np.load(epsilons_per_feature["glob"]).transpose()
                ).to(self.device),
                "cpf": torch.Tensor(
                    np.load(epsilons_per_feature["cpf"]).transpose()
                ).to(self.device),
                "npf": torch.Tensor(
                    np.load(epsilons_per_feature["npf"]).transpose()
                ).to(self.device),
                "vtx": torch.Tensor(
                    np.load(epsilons_per_feature["vtx"]).transpose()
                ).to(self.device),
            }

            while self.trainedepoches < nepochs:
                # this can change from epoch to epoch
                # calculate steps for this epoch
                # feed info below
                traingen.prepareNextEpoch()
                valgen.prepareNextEpoch()

                nbatches_train = (
                    traingen.getNBatches()
                )  # might have changed due to shuffeling
                nbatches_val = valgen.getNBatches()

                train_generator = traingen.feedNumpyData()
                val_generator = valgen.feedNumpyData()

                print(">>>> epoch", self.trainedepoches, "/", nepochs)
                print("training batches: ", nbatches_train)
                print("validation batches: ", nbatches_val)

                with tqdm(total=nbatches_train) as epoch_pbar:
                    epoch_pbar.set_description(f"Epoch {self.trainedepoches + 1}")

                    self.model.train()
                    for param_group in self.optimizer.param_groups:
                        print("/n Learning rate = " + str(param_group["lr"]) + " /n")
                    train_loss = train_loop(
                        train_generator,
                        nbatches_train,
                        self.model,
                        self.criterion,
                        self.optimizer,
                        self.device,
                        self.trainedepoches,
                        epoch_pbar,
                        attack,
                        att_magnitude,
                        restrict_impact,
                        epsilon_factors,
                        pgd_loops,
                        acc_loss=0,
                    )

                    self.scheduler.step()

                    self.model.eval()
                    val_loss = val_loop(
                        val_generator,
                        nbatches_val,
                        self.model,
                        self.criterion,
                        self.device,
                        self.trainedepoches,
                    )

                    self.trainedepoches += 1

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.saveModel(
                            self.model,
                            self.optimizer,
                            self.trainedepoches,
                            self.scheduler,
                            self.best_loss,
                            train_loss,
                            val_loss,
                            is_best=True,
                        )

                    self.saveModel(
                        self.model,
                        self.optimizer,
                        self.trainedepoches,
                        self.scheduler,
                        self.best_loss,
                        train_loss,
                        val_loss,
                        is_best=False,
                    )

                traingen.shuffleFileList()
