from subprocess import call
import numpy as np
import pickle
import time as t
import datetime
import os
from shutil import copyfile,copy,move
import easygui

# Select the Odorant, Odor Delivery Protocol, Locust Model
odor_path = easygui.fileopenbox(msg='Open Odor File',title='Odor Browser',default='/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/*.odor',filetypes=['*.odor'])
protocol_path = easygui.fileopenbox(msg='Open Protocol File',title='Odor Protocol Browser',default='/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odor Protocols/*.protocol',filetypes=['*.protocol'])
locust_path = easygui.fileopenbox(msg='Open Locust File',title='Locust Browser',default='/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Locusts/*.locust',filetypes=['*.locust'])

# Get Experiment Date metadata
dt = datetime.datetime.now()

# Generate Metadata File
meta_file = np.array([odor_path,protocol_path,locust_path])

print("Metadata Acquired. Starting Simulation.")

# Start Timer
start = t.time()

# Start Receptor Layer Processing
call(['python', 'receptorLayer.py', odor_path, protocol_path, locust_path])
