import numpy as np
import pickle

data = {}
data['duration'] = 6000
data['n_split'] = int(data['duration']/1000)
data['resolution'] = 0.01
data['n_n'] = 120
data['PNPN'] = 0.0
data['PNLN'] = 0.5
data['LNLN'] = 0.5
data['LNPN'] = 0.5
data['achmat'] = np.zeros((data['n_n'],data['n_n']))
data['gabamat'] = np.zeros((data['n_n'],data['n_n']))

n_n = data['n_n']
l_n = int(0.25*n_n)
p_n = int(0.75*n_n)

ach_mat = np.zeros((n_n,n_n))
ach_mat[p_n:,:p_n] = np.random.choice([0.,1.],size=(l_n,p_n),p=(1-data['PNLN'],data['PNLN'])) # PN->LN
ach_mat[:p_n,:p_n] = np.random.choice([0.,1.],size=(p_n,p_n),p=(1-data['PNPN'],data['PNPN'])) # PN->PN
np.fill_diagonal(ach_mat,0.)
data['achmat'] = ach_mat

gaba_mat = np.zeros((n_n,n_n))
gaba_mat[:p_n,p_n:] = np.random.choice([0.,1.],size=(p_n,l_n),p=(1-data['LNPN'],data['LNPN'])) # LN->PN
gaba_mat[p_n:,p_n:] = np.random.choice([0.,1.],size=(l_n,l_n),p=(1-data['LNLN'],data['LNLN'])) # LN->LN
np.fill_diagonal(gaba_mat,0.)
data['gabamat'] = gaba_mat

with open('model.pkl', 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
