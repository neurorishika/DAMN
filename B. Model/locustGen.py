import numpy as np
import pickle
import easygui
from datetime import datetime

dt = datetime.now()

data = {}

### Olfactory Receptor Neuron (Layer 1) ###

data['ORN_types'] = 100
data['ORN_replicates'] = 10
data['baseline'] = 0.02
data['latency'] = (0,200)
data['t_rise'] = (0,600)
data['t_fall'] = (0,1200)
data['tuning'] = (1,15)
data['a1'] = 15
data['a2'] = 0.8
data['inh_threshold'] = 60
data['f_sharp'] = 0.5
data['adaptation_extent'] = (0.1,1)
data['t_adaptation'] = (0,1200)

### Antennal Lobe (Layer 2) ###

data['AL_n'] = 120
data['PNPN'] = 0.0
data['PNLN'] = 0.5
data['LNLN'] = 0.5
data['LNPN'] = 0.5

l_n = int(0.25*data['AL_n'])
p_n = int(0.75*data['AL_n'])

### Layer 1 -> Layer 2 Connectivity ###

data['ORN-AL'] = np.zeros((data['ORN_types']*data['ORN_replicates'],data['AL_n']))
data['f_ORN-PN'] = 0.05
data['f_ORN-LN'] = 0.70

nc_PN = int(data['ORN_types']*data['f_ORN-PN'])
nc_LN = int(data['ORN_types']*data['ORN_replicates']*data['f_ORN-LN'])

pnc = []
for i in range(90):
    x = [1]*nc_PN+[0]*(100-nc_PN)
    np.random.shuffle(x)
    pnc.append(x)
data['ORN-AL'][:,:p_n] = np.array(list(np.array(pnc).T)*10)

lnc = []
for i in range(30):
    x = [1]*nc_LN+[0]*(1000-nc_LN)
    np.random.shuffle(x)
    lnc.append(x)
data['ORN-AL'][:,p_n:] = np.array(lnc).T

### Within Layer 2 Inter-Connectivity ###

data['achmat'] = np.zeros((data['AL_n'],data['AL_n']))
data['gabamat'] = np.zeros((data['AL_n'],data['AL_n']))

ach_mat = np.zeros((data['AL_n'],data['AL_n']))
ach_mat[p_n:,:p_n] = np.random.choice([0.,1.],size=(l_n,p_n),p=(1-data['PNLN'],data['PNLN'])) # PN->LN
ach_mat[:p_n,:p_n] = np.random.choice([0.,1.],size=(p_n,p_n),p=(1-data['PNPN'],data['PNPN'])) # PN->PN
np.fill_diagonal(ach_mat,0.)
data['achmat'] = ach_mat

gaba_mat = np.zeros((data['AL_n'],data['AL_n']))
gaba_mat[:p_n,p_n:] = np.random.choice([0.,1.],size=(p_n,l_n),p=(1-data['LNPN'],data['LNPN'])) # LN->PN
gaba_mat[p_n:,p_n:] = np.random.choice([0.,1.],size=(l_n,l_n),p=(1-data['LNLN'],data['LNLN'])) # LN->LN
np.fill_diagonal(gaba_mat,0.)
data['gabamat'] = gaba_mat

locust_path = easygui.filesavebox(msg='Save Locust File',title='Locust Browser',default='{}.locust'.format(dt.strftime("Locust_%d%m%Y_%H%M")),filetypes=['*.locust'])
with open(locust_path, 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
