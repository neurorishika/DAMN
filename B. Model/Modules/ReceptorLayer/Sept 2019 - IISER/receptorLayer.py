import numpy as np
import matplotlib.pyplot as plt
import polarTools as pt
import easygui
import pickle
import sys

print("Welcome to the ORNs !!!")

# Select the Odorant, Odor Delivery Protocol, Locust Model
odor_path = sys.argv[1]
protocol_path = sys.argv[2]
locust_path = sys.argv[3]

# Load the Odorant, Odor Delivery Protocol, Locust Model
with open(odor_path, 'rb') as fp:
    odor = pickle.load(fp)
with open(protocol_path, 'rb') as fp:
    protocol = pickle.load(fp)
with open(locust_path, 'rb') as fp:
    locust = pickle.load(fp)

# Define ORN Response Generator
def generate_orn(orn_number,duration,resolution,odorVec,odorStart,odorEnd): # Function to generate single ORN Trace
    
    baseline = 4/locust['peak_firing'] # Baseline Firing Rate Ratio
    trace = baseline*np.ones(int(duration/resolution)) # Set Baseline activity for the Protocol Duration
    rec_field = pt.generateUniform(1,odor['dim_odorspace'],seed=int(locust['rec_seeds'][orn_number])) # Receptive Field of ORNs in Odor Space
    
    latency = locust['latency'][orn_number] # Latency of Response to Odor Presentation
    t_rise = locust['t_rise'][orn_number] # Time to Rise to Peak
    t_fall = locust['t_fall'][orn_number] # Response Decay Time
    tuning = locust['tuning'][orn_number]/3 # Odor Tuning-width / Sensitivity
    
    def sigmoid(x,a1=locust['a1'],a2=locust['a2']):	# Sigmoid for Response
        return 1/(1+np.exp(-a1*(x-a2)))
    
    odorMag = np.linalg.norm(odorVec) # Odor Concentration
    cosSim = np.dot(odorVec,rec_field)/(np.linalg.norm(odorVec)*np.linalg.norm(rec_field)) # Cosine Similarity wrt Odor

    if np.arccos(cosSim) < np.deg2rad(locust['inh_threshold']):	# Minimum Response Threshhold
        res_strength = (1-baseline)*sigmoid(odorMag*np.cos(np.arccos(cosSim)/2)**tuning)
    else:
        res_strength = -baseline*odorMag
    
    if locust['f_sharp'][orn_number]:
        # Generate Sharp Trace
        rise = np.arange(0,t_rise/2,resolution)
        rise = baseline+res_strength*2*np.exp(1)/t_rise*rise*np.exp(-2*rise/t_rise)
        riseStartIndex = int((odorStart+latency)/resolution)
        riseEndIndex = riseStartIndex+rise.shape[0]
        trace[riseStartIndex:riseEndIndex] = rise
        peak = rise[-1]
        fall = np.linspace(0,duration-riseEndIndex*resolution,trace.shape[0]-riseEndIndex)
        fall = (peak-baseline)*np.exp(-fall/t_fall)+baseline
        fallStartIndex = riseEndIndex
        trace[fallStartIndex:] = fall    
    else:
        # Generate Broad Trace
        rise = np.arange(0,t_rise,resolution)
        rise = baseline+res_strength*np.exp(1)/t_rise*rise*np.exp(-rise/t_rise)
        riseStartIndex = int((odorStart+latency)/resolution)
        riseEndIndex = int((odorStart+latency)/resolution)+rise.shape[0]
        trace[riseStartIndex:riseEndIndex] = rise
        peak_1 = rise[-1]
        adaptation_rate = locust['adaptation_extent'][orn_number] # Amplitude of Adaptation-related Decay
        t_adaptation = locust['t_adaptation'][orn_number] # Odor Adaptation Time
        adaptation = np.arange(0,np.max([1,int(odorEnd/resolution)-riseEndIndex])*resolution,resolution)
        adaptation = (peak_1-(adaptation_rate*res_strength+baseline))*np.exp(-adaptation/t_adaptation)+(adaptation_rate*res_strength+baseline)
        adaptationStartIndex = riseEndIndex
        adaptationEndIndex = adaptationStartIndex+adaptation.shape[0]
        trace[adaptationStartIndex:adaptationEndIndex] = adaptation
        peak_2 = adaptation[-1]
        fall = np.arange(0,(trace.shape[0]-adaptationEndIndex)*resolution,resolution)
        fall = (peak_2-baseline)*np.exp(-fall/t_fall) + baseline
        fallStartIndex = adaptationEndIndex
        trace[fallStartIndex:] = fall
    
    trace = trace*locust['peak_firing'] # Scale to Peak Firing Rate
    
    return trace

# Generate Odor Response

print("Generating ORN Responses...")

orns = []
for i in range(locust['ORN_types']): # Generate ORN types
    orns.append(generate_orn(i,protocol['duration'],protocol['resolution'],odor['odor_vector'],protocol['odor_start'],protocol['odor_start']+protocol['odor_duration']))
    print('{}/{} ORN Types Completed'.format(i+1,locust['ORN_types']), end = '\r')

orns = np.array(orns*locust['ORN_replicates'])

print("Generation Complete. Plotting.")

# Plot the ORN Response
plt.figure()
#order = np.argsort(orns[:100,:].max(axis=1))
#np.save("order",order)
plt.rcParams.update({'font.size': 22})
order = np.load("order.npy")
plt.imshow(orns[order[::-1],::100], aspect='auto')
plt.clim(0,locust['peak_firing'])
plt.colorbar()
#plt.xlabel('Time (in ms)')
#plt.ylabel('Neuron Number')
#plt.title('ORN Response')
plt.savefig(odor_path.split('/')[-1]+"_"+'ORN Response.png',transparent=True)

# Plot the ORN Traces
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111, projection='3d')
ax.patch.set_alpha(0.0)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
order = np.argsort(orns[:100,:].mean(axis=1))
color=cm.RdBu(np.arange(0,order[::-1].shape[0]))
for n,i in enumerate(order[::-1]):
    ax.plot(5*n*np.ones(orns[i,::100].shape),np.arange(0,orns[i,::100].shape[0]),orns[i,::100],alpha=1)
#plt.ylabel('Time (in ms)')
#plt.xlabel('Cell Type')
#plt.title('ORN Traces')
ax.grid(False)
plt.axis('off')
plt.savefig(odor_path.split('/')[-1]+"_"+'ORN Traces.png')

# Plot EAD
plt.figure()
plt.plot(orns.mean(axis=0))
plt.xlabel('Time (in ms)')
plt.ylabel('Mean Firing Rate')
#plt.title('EAG Response')
plt.axis('off')
plt.savefig(odor_path.split('/')[-1]+"_"+'EAG Response.png',transparent=True)

# Save ORN Data
np.save(odor_path.split('/')[-1]+"_"+"current_input",orns[:,::100])


print("Generating Antennal Input...")

ORN_Output = np.matmul(orns.T,locust['ORN-AL']).T

p_n = int(0.75*locust['AL_n'])

PN_scale = 1/ORN_Output[:p_n,:].max() # PN Scaling Factor
LN_scale = 1/ORN_Output[p_n:,:].max() # LN Scaling Factor

# Scale ORN Output to AL Input
ORN_Output[:90,:] = (ORN_Output[:p_n,:] * PN_scale)
ORN_Output[90:,:] = (ORN_Output[p_n:,:] * LN_scale)

ORN_Output = ORN_Output + ORN_Output*locust['random_noise_level']*np.random.normal(size=ORN_Output.shape)

print("Generation Complete")

plt.rcParams.update({'font.size': 0})

# Plot PN Current 
plt.figure()
plt.plot(ORN_Output[:p_n,::100].T,alpha=0.7)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.ylim([0,1.5])
#plt.xlabel('Time (in ms)')
#plt.ylabel('PN Current Input')
plt.savefig('PN Current.png',transparent=True)

# Plot LN Current 
plt.figure()
plt.plot(ORN_Output[p_n:,::100].T,alpha=0.7)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
#plt.xlabel('Time (in ms)')
#plt.ylabel('LN Current Input')
plt.ylim([0,1.5])
plt.savefig('LN Current.png',transparent=True)

# Plot Overall Current
plt.figure()
order = np.concatenate((np.argsort(ORN_Output[:p_n,:].max(axis=1))[::-1],p_n+np.argsort(ORN_Output[p_n:,:].max(axis=1))[::-1]))
plt.imshow(ORN_Output[order,::100], aspect='auto')
plt.colorbar()
plt.xlabel('Time (in ms)')
plt.ylabel('Neuron Number')
plt.savefig('AL Input Current.png',transparent=True)

# Save Current Input
np.save('current_input',ORN_Output)

print("'Information has been transferred to the Antennal Lobe. Thank you for using our services.' - ORNs")


