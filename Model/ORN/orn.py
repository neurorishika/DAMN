import numpy as np
import matplotlib.pyplot as plt

## Implementation of Raman et. al. 2010 ORN Model ##

l_odorspace = 2		# Number of Parameters that define Odorants

# More parameters generally means a sparser response by ORNs 

odor = np.random.uniform(0,1,l_odorspace) # Odorant Identity in Odor Space

# The Magnitude of Odor Vector is proportional to Concentration

def generate_orn(duration,resolution,odorVec,odorStart,odorEnd): # Function to generate single ORN Trace
    
    trace = np.zeros(int(duration/resolution))
    rec_field = np.random.uniform(0,1,l_odorspace) # Receptive Field of ORNs in Odor Space
    
    latency = np.random.uniform(0,200) # Latency of Response to Odor Presentation
    t_rise = np.random.uniform(0,600) # Time to Rise to Peak
    t_fall = np.random.uniform(0,1200) # Response Decay Time
    tuning = np.random.uniform(1,15) # Odor Tuning-width / Sensitivity
    
    def sigmoid(x,a1=15,a2=0.8):	# Sigmoid for Response
        return 1/(1+np.exp(-a1*(x-a2)))
    
    odorMag = np.linalg.norm(odorVec) # Odor Concentration
    cosSim = np.dot(odorVec,rec_field)/(np.linalg.norm(odorVec)*np.linalg.norm(rec_field)) # Cosine Similarity wrt Odor
    

    if np.arccos(cosSim) < 1.43117:	# Minimum Response Threshhold
        res_strength = sigmoid(odorMag*cosSim**tuning)
    else:
        res_strength = 0
    
    
    if np.random.uniform()<0.5:
        
        response_type='sharp' # Sharp Response to Odor
        
        # Generate Sharp Trace

        rise = np.arange(0,t_rise/2,resolution)
        rise = res_strength*2*np.exp(1)/t_rise*rise*np.exp(-2*rise/t_rise)
        
        riseStartIndex = int((odorStart+latency)/resolution)
        riseEndIndex = riseStartIndex+rise.shape[0]
        
        trace[riseStartIndex:riseEndIndex] = rise
        
        peak = rise[-1]
        
        fall = np.linspace(0,duration-riseEndIndex*resolution,trace.shape[0]-riseEndIndex)
        fall = peak*np.exp(-fall/t_fall)
        
        fallStartIndex = riseEndIndex
        trace[fallStartIndex:] = fall
    
    else:
        
        response_type='broad' # Broad Response to Odor
        
        # Generate Broad Trace

        rise = np.arange(0,t_rise,resolution)
        rise = res_strength*np.exp(1)/t_rise*rise*np.exp(-rise/t_rise)
        
        riseStartIndex = int((odorStart+latency)/resolution)
        riseEndIndex = int((odorStart+latency)/resolution)+rise.shape[0]
        
        trace[riseStartIndex:riseEndIndex] = rise
        
        peak_1 = rise[-1]
        
        adaptation_rate = np.random.uniform(0.1,1) # Amplitude of Adaptation-related Decay
        
        adaptation = np.arange(0,(int(odorEnd/resolution)-riseEndIndex)*resolution,resolution)

        if res_strength>0:
            adaptation = np.linspace(peak_1,adaptation_rate*res_strength,adaptation.shape[0],endpoint=False)
        else:
            adaptation = np.zeros(adaptation.shape)

        adaptationStartIndex = riseEndIndex
        adaptationEndIndex = adaptationStartIndex+adaptation.shape[0]
        
        trace[adaptationStartIndex:adaptationEndIndex] = adaptation
        
        peak_2 = adaptation[-1]
        
        fall = np.arange(0,(trace.shape[0]-adaptationEndIndex)*resolution,resolution)
        fall = peak_2*np.exp(-fall/t_fall)
        
        fallStartIndex = adaptationEndIndex
        
        trace[fallStartIndex:] = fall
    
    time = np.arange(0,duration,resolution)
    
    return time,trace

# Generate Odor Response

orns = []

for i in range(100):	# Generate 100 ORN types
    time,trace = generate_orn(5000,0.01,odor,500,1500)
    orns.append(trace)

orns = np.array(orns*10) # Make 10 replicates of each ORN

# Plot ORN Responses
plt.figure()
order = np.argsort(orns.mean(axis=1))
plt.imshow(orns[order[::-1],::100], aspect='auto')
plt.colorbar()
plt.xlabel('Time (in ms)')
plt.ylabel('Neuron Number')
plt.title('ORN Response')
plt.savefig('orn_response.png')

# Plot EAD
plt.figure()
plt.plot(np.matmul(orns.T,np.ones((1000,1))))
plt.xlabel('Time (in ms)')
plt.ylabel('Firing Rate')
plt.title('EAG Response')
plt.savefig('eag_response.png')

# Generate ORN-AL Connectivity (1000 ORNs to 90 PNs and 30 LNs)

ORN_AL = np.zeros((1000,120))

p_PN = 0.05
p_LN = 0.7

ORN_AL[:,:90] = np.random.choice([0,1],size=(1000,90),p=[1-p_PN,p_PN])
ORN_AL[:,90:] = np.random.choice([0,1],size=(1000,30),p=[1-p_LN,p_LN])

# Generate Antennal Output

ORN_Output = np.matmul(orns.T,ORN_AL).T

# Plot PN Response
plt.figure()
order = np.argsort(ORN_Output[:90,:].mean(axis=1))
plt.imshow(ORN_Output[order[::-1],::100], aspect='auto')
plt.xlabel('Time (in ms)')
plt.ylabel('PN Number')
plt.title('PN Response Profile')
plt.savefig('pn_response.png')

# Plot LN Response
plt.figure()
order = np.argsort(ORN_Output[90:,:].mean(axis=1))
plt.imshow(ORN_Output[90+order[::-1],::100], aspect='auto')
plt.xlabel('Time (in ms)')
plt.ylabel('LN Number')
plt.title('LN Response Profile')
plt.savefig('ln_response.png')

baseline = 2 # Baseline Current Input

PN_current = 6 - baseline # Effective Current Input to PNs
LN_current = 7 - baseline # Effective Current Input to LNs

PN_scale = PN_current/ORN_Output[:90,50000:150000].mean() # PN Scaling Factor
LN_scale = LN_current/ORN_Output[90:,50000:150000].mean() # LN Scaling Factor

# Scale ORN Output to AL Input
ORN_Output[:90,:] = baseline + (ORN_Output[:90,:] * PN_scale)
ORN_Output[90:,:] = baseline + (ORN_Output[90:,:] * LN_scale)

# Plot AL Input
plt.figure()
plt.imshow(ORN_Output[:,::100], aspect='auto')
plt.colorbar()
plt.xlabel('Time (in ms)')
plt.ylabel('Neuron Number')
plt.title('AL Current Input')
plt.savefig('current_input.png')

# Save Current Input
np.save('current_input',ORN_Output)

