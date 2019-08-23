import numpy as np
import pickle
import easygui

data = {}
data['duration'] = 6000
data['n_split'] = int(data['duration']/1000)
data['resolution'] = 0.01
data['n_odor_pulse'] = 1
data['odor_start'] = 500
data['pulse_duration'] = 1000
data['pulse_delay'] = 500
data['odors'] = ['0_1']

protocol_path = easygui.filesavebox(msg='Save Protocol File',title='Protocol Browser',default='Dur_{}_Pulse_{}_Odor_{}.protocol'.format(data['duration'],data['pulse_duration'],data['odors']),filetypes=['*.protocol'])
with open(protocol_path, 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
