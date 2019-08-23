from subprocess import call
import numpy as np
import pickle
import time as t
import datetime
import os
from shutil import copyfile,move
import easygui

locust_path = easygui.fileopenbox(msg='Open Locust File',title='Locust Browser',default='*.locust',filetypes=['*.locust','*.pkl'])
protocol_path = easygui.fileopenbox(msg='Open Protocol File',title='Odor Protocol Browser',default='*.protocol',filetypes=['*.protocol','*.pkl'])

dt = datetime.datetime.now()

start = t.time()

with open(protocol_path, 'rb') as fp:
    data = pickle.load(fp)

time = np.split(np.arange(0,data['duration'],data['resolution']),data['n_split'])

for n,i in enumerate(time):
    if n>0:
        time[n] = np.append(i[0]-0.01,i)

np.save("time",time)


folder = "/home/iiser/Collins-Saptarshi 2019b/DAMN/Model/Simulation Data/"+dt.strftime("%d%B%Y_%H%M")

if not os.path.exists(folder):
    os.makedirs(folder)

for i in range(data['n_split']):
    call(['python','run.py',str(i), locust_path,protocol_path])

os.remove('state_vector.npy')
os.remove('time.npy')

print("Simulation Completed. Time taken: {:0.2f}".format(t.time()-start))

call(['python','analyse.py'])

for f in filter(lambda v: (".pkl" in v) or (".npy" in v) or (".png" in v),os.listdir()):
    move(os.getcwd()+"/"+f, folder+'/'+f)


