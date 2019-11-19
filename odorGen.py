import numpy as np
import pickle
import easygui
import polarTools as pt

data = {}
data['dim_odorspace'] = 2
data['odor_concentration'] = 1
data['reference_odor'] = np.zeros(data['dim_odorspace'])
data['reference_odor'][0] = data['odor_concentration']

data['odor_type'] = input("Enter Odor Type: ")

data['seed'] = np.random.randint(0,1000000000)
np.random.seed(data['seed'])

if data['odor_type'] == 'reference':
    data['odor_vector'] = data['reference_odor']
    print(np.dot(data['odor_vector'],data['reference_odor'])/np.linalg.norm(data['reference_odor'])/np.linalg.norm(data['odor_vector']))
    data['odor_angle'] = np.rad2deg(np.arccos(np.clip(np.dot(data['odor_vector'],data['reference_odor'])/np.linalg.norm(data['reference_odor'])/np.linalg.norm(data['odor_vector']),-1,1)))
elif data['odor_type'] == 'no_odor':
    data['odor_vector'] = np.zeros(data['dim_odorspace'])
    data['odor_angle'] = 0
elif data['odor_type'] == 'random':
    
    data['odor_vector'] = pt.generateUniform(data['odor_concentration'],dimension=data['dim_odorspace'])
    data['odor_angle'] = np.rad2deg(np.arccos(np.dot(data['odor_vector'],data['reference_odor'])/np.linalg.norm(data['reference_odor'])/np.linalg.norm(data['odor_vector'])))
elif data['dim_odorspace'] == 2:
    data['odor_type'] = int(data['odor_type'])
    if np.random.normal() > 0:
        theta = np.deg2rad(data['odor_type'])
    else:
        theta = -np.deg2rad(data['odor_type'])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    data['odor_vector'] = np.matmul(R,data['reference_odor'].reshape(-1,1)).flatten()
    data['odor_angle'] = np.rad2deg(np.arccos(np.dot(data['odor_vector'],data['reference_odor'])/np.linalg.norm(data['reference_odor'])/np.linalg.norm(data['odor_vector'])))
elif data['dim_odorspace'] == 3:
    data['odor_type'] = int(data['odor_type'])
    randVec = pt.generateUniform(1,dimension=data['dim_odorspace'])
    crossVec = np.cross(data['reference_odor'],randVec)
    crossVec = crossVec/np.linalg.norm(crossVec)
    s,r = np.random.uniform(0,1),np.random.uniform(0,1)
    h = np.cos(np.deg2rad(data['odor_type']))
    z = h+(1-h)*r
    phi = 2*np.pi*s
    sin = np.sqrt(1-z*z)
    x = np.cos(phi)*sin
    y = np.sin(phi)*sin
    data['odor_vector'] = randVec * x + crossVec * y + data['reference_odor'] * z
    data['odor_angle'] = np.rad2deg(np.arccos(np.dot(data['odor_vector'],data['reference_odor'])/np.linalg.norm(data['reference_odor'])/np.linalg.norm(data['odor_vector'])))
    
if data['odor_type'] == 'random':
    odor_path = easygui.filesavebox(msg='Save Odor File',title='Odor Browser',default='/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/Conc_{}_Angle_{:0.2f}_Type_{}_Dimension_{}_seed_{}.odor'.format(data['odor_concentration'],data['odor_angle'],data['odor_type'],data['dim_odorspace'],data['seed']),filetypes=['*.odor'])
else:
    odor_path = easygui.filesavebox(msg='Save Odor File',title='Odor Browser',default='/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odors/Conc_{}_Angle_{:0.2f}_Type_{}_Dimension_{}.odor'.format(data['odor_concentration'],data['odor_angle'],data['odor_type'],data['dim_odorspace']),filetypes=['*.odor'])

with open(odor_path, 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
