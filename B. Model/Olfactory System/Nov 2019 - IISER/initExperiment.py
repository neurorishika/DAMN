from subprocess import call
import numpy as np
import pickle
import time as t
import datetime
import os
from shutil import copyfile,copy,move
import easygui
import sys

# Select the Odorant, Odor Delivery Protocol, Locust Model
odor_path = sys.argv[1]# easygui.fileopenbox(msg='Open Odor File',title='Odor Browser',default='/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/*.odor',filetypes=['*.odor'])
protocol_path = sys.argv[2]# easygui.fileopenbox(msg='Open Protocol File',title='Odor Protocol Browser',default='/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odor Protocols/*.protocol',filetypes=['*.protocol'])
locust_path = sys.argv[3]# easygui.fileopenbox(msg='Open Locust File',title='Locust Browser',default='/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Locusts/*.locust',filetypes=['*.locust'])

# Get Experiment Date metadata
dt = datetime.datetime.now()

# Generate Metadata File
meta_file = np.array([odor_path,protocol_path,locust_path])

print("Metadata Acquired. Starting Simulation.")

# Start Timer
start = t.time()

# Start Receptor Layer Processing
call(['python', 'receptorLayer.py', odor_path, protocol_path, locust_path])

# Load Protocol data
with open(protocol_path, 'rb') as fp:
    data = pickle.load(fp)

# Generate Batch-time for AL Simulation
time = np.split(np.arange(0,data['duration'],data['resolution']),data['n_split'])
for n,i in enumerate(time):
    if n>0:
        time[n] = np.append(i[0]-0.01,i)
np.save("time",time)

# Start Antennal Lobe Processing

print("Welcome to the AL !!!")

for i in range(data['n_split']):
    call(['python','antennalLobe.py',str(i), locust_path,protocol_path])

os.remove('state_vector.npy')
os.remove('time.npy')

print("Simulation Completed. Time taken: {:0.2f}".format(t.time()-start))

print("'Thank you for using our services.'-AL")

# Start AL Analysis
call(['python','analyse.py'])

filename = odor_path.split('/')[-1].split('.')[0]+"_"+protocol_path.split('/')[-1].split('.')[0]+"_"+locust_path.split('/')[-1].split('.')[0]+"_"+sys.argv[4]

# Generate Experiment Directory 
folder = "/home/iiser/Collins-Saptarshi 2019b/DAMN/B. Model/Simulation Data/EndSem/"+filename
if not os.path.exists(folder):
    os.makedirs(folder)

# Move Data to Experiment Directory
for f in filter(lambda v: (".pkl" in v) or (".npy" in v) or (".png" in v),os.listdir()):
    move(os.getcwd()+"/"+f, folder+'/'+f)

# Copy metadata files
copy(odor_path,folder)
copy(protocol_path,folder)
copy(locust_path,folder)
copy(os.getcwd()+"/antennalLobe.py",folder)
