from pytorch_deepjet_transformer import DeepJetTransformer
from pytorch_first_try import training_base
from pytorch_deepjet_run2 import *
from pytorch_ranger import Ranger
from pytorch_deepjet import *
import torch.nn as nn
import numpy as np
import torch
import os

print(f"This process has the PID {os.getpid()} .")

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


num_epochs = 60

lr_epochs = max(1, int(num_epochs * 0.3))
lr_rate = 0.01 ** (1.0 / lr_epochs)
mil = list(range(num_epochs - lr_epochs, num_epochs))

model = DeepJet_Run2(
    num_classes=6
)  # DeepJet(num_classes = 6) #DeepJetTransformer(num_classes = 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = cross_entropy_one_hot
optimizer = Ranger(
    model.parameters(), lr=5e-3
)  # torch.optim.Adam(model.parameters(), lr = 0.003, eps = 1e-07)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=mil, gamma=lr_rate
)

train = training_base(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    testrun=False,
)

train.train_data.maxFilesOpen = 1

attack = "PGD"
att_magnitude = 0.05
restrict_impact = -1
pgd_loops = 5

model, history = train.trainModel(
    nepochs=num_epochs + lr_epochs,
    batchsize=4000,
    attack=attack,
    att_magnitude=att_magnitude,
    restrict_impact=restrict_impact,
    pgd_loops=pgd_loops,
)

print(f"Finished process {os.getpid()} .")
