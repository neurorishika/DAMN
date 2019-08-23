from subprocess import call
import numpy as np
import pickle
import time as t

start = t.time()

with open('model.pkl', 'rb') as fp:
    data = pickle.load(fp)

time = np.split(np.arange(0,data['duration'],data['resolution']),data['n_split'])

for n,i in enumerate(time):
    if n>0:
        time[n] = np.append(i[0]-0.01,i)

np.save("time",time)

for i in range(data['n_split']):
    call(['python','run.py',str(i)])

print("Simulation Completed. Time taken: {:0.2f}".format(t.time()-start))
