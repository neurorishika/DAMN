import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import os

n_splits = len(list(filter(lambda v: 'batch' in v, os.listdir())))
n_batch = 1

overall_state = []

# Iterate over the generated output files
for n,i in enumerate(["batch"+str(x+1) for x in range(n_splits)]):
    for m,j in enumerate(["_part_"+str(x+1)+".npy" for x in range(n_batch)]):
        # Since the first element in the series was the last output, we remove them
        if n>0 and m>0:
            overall_state.append(np.load(i+j)[1:,:120])
        else:
            overall_state.append(np.load(i+j)[:,:120])
        print(i+j)

# Concatenate all the matrix to get a single state matrix
overall_state = np.concatenate(overall_state)

print("Concatenated")


current_input = np.load("current_input.npy")

orderPN = np.argsort(current_input[:90,:].max(axis=1))[::-1]
orderLN = 90+np.argsort(current_input[90:120,:].max(axis=1))[::-1]

order = np.concatenate((orderPN,orderLN),axis=None)

plt.figure()
plt.imshow(current_input[order,::100], aspect='auto')
plt.colorbar()
plt.xlabel('Time (in ms)')
plt.ylabel('Neuron Number')
plt.title('AL Current Input')
plt.savefig('current_input.png')

plt.figure(figsize=(12,6))
    
sns.heatmap(overall_state[::100,order].T,xticklabels=100,yticklabels=5)

plt.xlabel("Time (in ms)")
plt.ylabel("Neuron Number")
plt.title("Voltage vs Time Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")

plt.figure(figsize=(20,3))
plt.plot(np.arange(0,overall_state.shape[0]*0.01,0.01),overall_state[:,:90].mean(axis=1))
plt.tight_layout()
plt.savefig("LFP.png")

y = overall_state[50000:450000,:90].mean(axis=1)
Y = np.fft.fft(y)
freq = np.fft.fftfreq(len(y), 0.01/1000)
ind = np.logical_and(freq>0,freq<100)

plt.figure(figsize=(12,6))
plt.plot( freq[ind], np.abs(Y)[ind])
plt.savefig("fourier.png")

