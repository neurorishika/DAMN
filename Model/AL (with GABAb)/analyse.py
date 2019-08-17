import numpy as np
import matplotlib.pyplot as plt

n_splits = int(input('Enter Number of Splits: ') or 1)
n_batch = 1

overall_state = []

# Iterate over the generated output files
for n,i in enumerate(["batch"+str(x+1) for x in range(n_splits)]):
    for m,j in enumerate(["_part_"+str(x+1)+".npy" for x in range(n_batch)]):
        print(i+j)
        # Since the first element in the series was the last output, we remove them
        if n>0 and m>0:
            overall_state.append(np.load(i+j)[1:,:])
        else:
            overall_state.append(np.load(i+j))

# Concatenate all the matrix to get a single state matrix
overall_state = np.concatenate(overall_state)

print("Concatenated")

plt.figure(figsize=(12,6))
    
sns.heatmap(overall_state[::100,:120].T,xticklabels=100,yticklabels=5,cmap='RdBu_r')

plt.xlabel("Time (in ms)")
plt.ylabel("Neuron Number")
plt.title("Voltage vs Time Heatmap")

plt.tight_layout()
plt.savefig("heatmap.png")
