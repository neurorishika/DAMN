import numpy as np
import pickle
import easygui

data = {}
data['duration'] = 8000
data['n_split'] = int(data['duration']/1000)
data['resolution'] = 0.01
data['odor_start'] = 500
data['odor_duration'] = 500

protocol_path = easygui.filesavebox(msg='Save Protocol File',title='Protocol Browser',default='Dur_{}_Start_{}_OdorDur_{}.protocol'.format(data['duration'],data['odor_start'],data['odor_duration']),filetypes=['*.protocol'])
with open(protocol_path, 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
