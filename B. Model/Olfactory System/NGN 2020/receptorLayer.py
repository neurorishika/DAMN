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
    
    baseline = locust['baseline_firing']*(1+0.2*np.random.normal())/locust['peak_firing'] # Baseline Firing Rate Ratio
    trace = baseline*np.ones(int(duration/resolution)) # Set Baseline activity for the Protocol Duration
    rec_field = pt.generateUniform(1,odor['dim_odorspace'],seed=int(locust['rec_seeds'][orn_number])) # Receptive Field of ORNs in Odor Space
    
    latency = locust['latency'][orn_number] # Latency of Response to Odor Presentation
    t_rise = locust['t_rise'][orn_number] # Time to Rise to Peak
    t_fall = locust['t_fall'][orn_number] # Response Decay Time
    tuning = locust['tuning'][orn_number]/2 # Odor Tuning-width / Sensitivity
    
    def sigmoid(x,a1=locust['a1'],a2=locust['a2']):	# Sigmoid for Response
        return 1/(1+np.exp(-a1*(x-a2)))
    
    odorMag = 0.85*np.linalg.norm(odorVec) # Odor Concentration
    cosSim = np.dot(odorVec,rec_field)/(np.linalg.norm(odorVec)*np.linalg.norm(rec_field)) # Cosine Similarity wrt Odor

    if np.arccos(cosSim) < np.deg2rad(locust['inh_threshold']):	# Minimum Response Threshhold
        res_strength = (1-baseline)*sigmoid(odorMag*np.cos(np.arccos(cosSim)/2)**tuning)
    else:
        res_strength = -baseline*np.linalg.norm(odorVec)
    
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
        adaptation = np.arange(0,(int(odorEnd/resolution)-riseEndIndex)*resolution,resolution)
        adaptation = (peak_1-(adaptation_rate*res_strength+baseline))*np.exp(-adaptation/t_adaptation)+(adaptation_rate*res_strength+baseline)
        adaptationStartIndex = riseEndIndex
        adaptationEndIndex = adaptationStartIndex+adaptation.shape[0]
        trace[adaptationStartIndex:adaptationEndIndex] = adaptation
        peak_2 = adaptation[-1]
        fall = np.arange(0,(trace.shape[0]-adaptationEndIndex)*resolution,resolution)
        fall = (peak_2-baseline)*np.exp(-fall/t_fall) + baseline
        fallStartIndex = adaptationEndIndex
        trace[fallStartIndex:] = fall
    
    trace = trace*locust['peak_firing']*5 # Scale to Peak Firing Rate
    
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
#order = np.argsort(orns.max(axis=1))
plt.imshow(orns[:100,::100], aspect='auto')
plt.colorbar()
plt.clim(0, 100)
plt.xlabel('Time (in ms)')
plt.ylabel('Neuron Number')
plt.title('ORN Response')
plt.savefig('ORN Response.png')

# Plot the ORN Traces
plt.figure()
order = np.argsort(orns.mean(axis=1))
plt.plot(orns[:,::100].T)
plt.xlabel('Time (in ms)')
plt.ylabel('Neuron Number')
plt.title('ORN Traces')
plt.savefig('ORN Traces.png')

# Plot EAD
plt.figure()
plt.plot(orns.mean(axis=0))
plt.xlabel('Time (in ms)')
plt.ylabel('Mean Firing Rate')
plt.title('EAG Response')
plt.savefig('EAG Response.png')

# Save ORN Data
np.save('ORN Firing Data',orns[:,::100])


def spike_generator(fr,resolution):
    resolution = resolution/1000
    spike = np.zeros(fr.shape)
    w= 0.5
    i = int(np.random.exponential(1/fr[0])/resolution)
    spike[i] = 1
    while True:
        t_next = int(np.max([w/fr[i]+(1-w)*np.random.exponential(1/fr[i]),0.003])/resolution)
        if i+t_next>=fr.shape[0]:
            break
        spike[i+t_next]= 1
        i=i+t_next
    return spike

orns_spike = np.zeros(orns.shape)
for i in range(orns.shape[0]):
    orns_spike[i,:] = spike_generator(orns[i,:],0.01)
    print('{}/{} ORN Spiking Completed'.format(i+1,locust['ORN_types']*locust['ORN_replicates']), end = '\r')
print()

# Generate Antennal Output

print("Generating Antennal Input...")

ORN_Output_s = np.matmul(orns_spike.T,locust['ORN-AL']).T

p_n = int(0.75*locust['AL_n'])

ORN_Output_current = np.zeros(ORN_Output_s.shape)
for i in range(ORN_Output_s.shape[0]):
    cfilter = 0.5*np.ones(30)
    ORN_Output_current[i,:] = np.convolve(ORN_Output_s[i,:], cfilter,'same')
    print('{}/{} Acetylcholine Concentration Integration Completed'.format(i+1,locust['AL_n']), end = '\r')
print()

plt.figure(figsize=(12,3))
order = np.concatenate((np.argsort(ORN_Output_current[:p_n,:].mean(axis=1))[::-1],p_n+np.argsort(ORN_Output_current[p_n:,:].mean(axis=1))[::-1]))
plt.imshow(ORN_Output_current[order,::100], aspect='auto')
plt.colorbar()
plt.xlabel('Time (in ms)')
plt.ylabel('Neuron Number')
plt.savefig('Acetylcholine Concentration.png')

ep=0.01
a = 10.0
b = 0.2

def f(o,t):
#    do = a*(1.0-o)*ORN_Output_current[:,int(t/ep)]/np.array([50]*90+[700]*30) - b*o
    do = a*(1.0-o)*ORN_Output_current[:,int(t/ep)]/np.array([50]*90+[700]*30) - b*o
    return do

time = np.arange(ORN_Output_current.shape[1])*ep
X = np.zeros(ORN_Output_current.shape)

X[:,0]= 0

for i in range(1,time.shape[0]):
    X[:,i] = X[:,i-1] + ep*f(X[:,i-1],time[i-1])
    if i%int(100/ep) == 0:
        print('{}s/{}s Acetylcholine Receptor Integration Completed'.format(i*ep,time.shape[0]*ep), end = '\r')
print()

print("Generation Complete")

# Plot PN Current 
plt.figure()
plt.plot(X[:p_n,::100].T)
plt.xlabel('Time (in ms)')
plt.ylabel('PN Current Input')
plt.savefig('PN Current.png')

# Plot LN Current 
plt.figure()
plt.plot(X[p_n:,::100].T)
plt.xlabel('Time (in ms)')
plt.ylabel('LN Current Input')
plt.savefig('LN Current.png')

# Plot Overall Current
plt.figure()
order = np.concatenate((np.argsort(X[:p_n,:].mean(axis=1))[::-1],p_n+np.argsort(X[p_n:,:].mean(axis=1))[::-1]))
plt.imshow(X[order,::100], aspect='auto')
plt.colorbar()
plt.xlabel('Time (in ms)')
plt.ylabel('Neuron Number')
plt.savefig('AL Input Current.png')

# Save Current Input
np.save('current_input',X)

print("'Information has been transferred to the Antennal Lobe. Thank you for using our services.' - ORNs")
