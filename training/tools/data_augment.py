import h5py
from mani_skill_learn.utils.fileio import load_h5_as_dict_array
import numpy as np

def rotx(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def flip(pcd):
    pcd['xyz'][:,:,1] = -1 * pcd['xyz'][:,:,1]
    return pcd

def subcenter(pcd):
    xyz_center = np.mean(pcd['xyz'], axis=1, keepdims=True)
    pcd['xyz'] -= xyz_center
    return pcd

def scale(pcd):
    pcd['xyz'] *= np.random.uniform(0.9,1.1)
    return pcd

def rotate(pcd):
    angleX = (np.random.random()*np.pi/9) - np.pi/18
    angleY = (np.random.random()*np.pi/9) - np.pi/18
    angleZ = (np.random.random()*np.pi/9) - np.pi/18
    pcd['xyz'] = np.dot(pcd['xyz'],np.transpose(rotx(angleX)))
    pcd['xyz'] = np.dot(pcd['xyz'],np.transpose(roty(angleY)))
    pcd['xyz'] = np.dot(pcd['xyz'],np.transpose(rotz(angleZ)))
    return pcd

def colorJitter(pcd):
    rgb_color = pcd['rgb'] #+ MEAN_COLOR_RGB
    rgb_color *= (1+0.4*np.random.random(3)-0.2) # brightness change for each channel
    rgb_color += (0.1*np.random.random(3)-0.05) # color shift for each channel
    rgb_color += np.expand_dims((0.05*np.random.random(pcd['rgb'].shape[1])-0.025), -1) # jittering on each pixel
    rgb_color = np.clip(rgb_color, 0, 1)
    # 20% gray scale
    random_idx = np.random.choice(rgb_color.shape[0], rgb_color.shape[0]//5, replace=False)
    rgb_color[random_idx] = np.stack([np.dot(rgb_color[random_idx],np.array([0.3,0.59,0.11])), np.dot(rgb_color[random_idx],np.array([0.3,0.59,0.11])), np.dot(rgb_color[random_idx],np.array([0.3,0.59,0.11]))], axis=-1)
    # randomly drop out 30% of the points' colors
    rgb_color *= np.expand_dims(np.random.random(pcd['rgb'].shape[1])>0.3,-1)
    pcd['rgb'] = np.maximum(rgb_color,0) ### Subtract mean color
    return pcd

def pcd_transform(pcd):
    p1 = colorJitter(pcd)
    p2 = rotate(p1)
    p3 = scale(p2)
    p4 = flip(p3)
    p5 = subcenter(p4)
    return p5
'''
path = "/data2/ManiSkill_Data/ShenHao/ManiSkill_Data/full_mani_skill_data/OpenCabinetDrawer/OpenCabinetDrawer_1000_link_0-v0.h5"
input_h5 = h5py.File(path, 'r')
obs = load_h5_as_dict_array(input_h5["traj_0"]['obs'])
pcd0 = obs['pointcloud']

pcd = pcd_transform(pcd0)
with open("/home/weikangwan/ManiSkill-Learn/full_mani_skill_data/point.txt", 'w') as f:
    fra = 10
    print(obs['state'])
    print(len(pcd['xyz']))
    N = len(pcd['xyz'][fra])
    for i in range(0,N):
        nda = pcd['xyz'][fra][i]  
        
        """    
        bl = pcd['seg'][fra][i]
        st = '0 0 0'
        if (bl[0]): st = '1 0 0'
        elif (bl[1]): st = '0 1 0'
        elif (bl[2]): st = '0 0 1'
        """
        color = pcd['rgb'][fra][i]
        f.write(' '.join(str(e) for e in nda) + ' ' + ' '.join(str(e) for e in color) + '\n')

'''