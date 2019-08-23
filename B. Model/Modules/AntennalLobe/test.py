from subprocess import call
import numpy as np
import pickle
import time as t
import datetime
import os
from shutil import copyfile,move

dt = datetime.datetime.now()

start = t.time()

with open('model.pkl', 'rb') as fp:
    data = pickle.load(fp)

folder = "/home/iiser/Collins-Saptarshi 2019b/DAMN/Model/Simulation Data/"+dt.strftime("%d%B%Y_%H%M")

if not os.path.exists(folder):
    os.makedirs(folder)

for f in filter(lambda v: (".pkl" in v) or (".npy" in v) or (".png" in v),os.listdir()):
    move(os.getcwd()+"/"+f, folder+'/'+f)
