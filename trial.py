#!/usr/bin/env python
# coding: utf-8

# In[173]:


import numpy as np
import pydiffmap as dm
import matplotlib.pyplot as plt 


# In[234]:


# we want to choose for every event, the best waveforms in order to lower the amount of data in the data set
# this function will get all waveforms for each event, will create a histogran of its values and choose the 3
# who have the smallest varience in values.


def varience_choose(waveforms,n):#the waveforms of one specific event , n is the amount of waveforms you want to get back
    length = len(waveforms) #the amount fo waveforms recieved per event
    if length <= n :
        return np.arange(length)
    vari = np.zeros(length)
    for i in range(length):
        vari[i] = np.var(waveforms[i])
    index = np.argpartition(vari,n)[:n] # smallest varience
    return index

def mean_varience_choose(waveforms,n):#the waveforms of one specific event , n is the amount of waveforms you want to get back
    length = len(waveforms) #the amount fo waveforms recieved per event
    if length <= n :
        return np.arange(length)
    vari = np.zeros(length)
    for i in range(length):
        vari[i] = np.var(waveforms[i])
        
    mean_var = np.average(vari)/length
    index = []
    for j in range(n):    
        ind = find_nearest(vari,mean_var)
        index.append(ind)
        vari[ind] = np.inf

    return np.array(index)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def knn_choose(waveforms,n):
    length = len(waveforms) #the amount fo waveforms recieved per event
    if length <= n :
        return np.arange(length)
    mean = (np.sum(waveforms,1))/np.size(waveforms,1)
    mean = np.expand_dims(mean,1)
    index = find_nearest2mean(mean,n)
    return index
    

def dm_choose(waveforms,n): # waveforms is a list ofwaveforms for the same event 
    length = len(waveforms) #the amount fo waveforms recieved per event
    if length <= n :
        return  np.arange(length)
    data = np.array(waveforms)
    embaded_data = DM(data)
    dm_indx = find_nearest2mean(embaded_data,n)
    return dm_indx
    

def DM(inputMatrix):
    n_evecs = 3
    epsilon = 'bgh'
    alpha = 0.5
    k = 49
    kernel_type = 'gaussian'
    metric = 'euclidean'
    bandwidth_normalize = False
    oos = 'nystroem'
    dim = 3
    my_kernel = dm.kernel.Kernel(kernel_type, epsilon, k, metric=metric)
    my_dmap = dm.diffusion_map.DiffusionMap(my_kernel, alpha, n_evecs, bandwidth_normalize=bandwidth_normalize, oos=oos)
    my_dmap.dmap = my_dmap.fit_transform(inputMatrix)
    dmap_embedded_data = my_dmap.dmap
    #print('embedded data:\n{}'.format(dmap_embedded_data))
    #print('size of embedded data:\n{}'.format(np.size(dmap_embedded_data)))
    #values = my_dmap.evals
    #dm.visualization.embedding_plot(my_dmap, dim, show=True)
    return dmap_embedded_data


def find_nearest2mean(data,N):
    k = np.shape(data)[0]
    mean = np.mean(data, axis=0)
    dist_array = np.zeros(k)
    for i in range(k):
        dist_array[i] = np.linalg.norm(mean - data[i,:])
    index = np.argpartition(dist_array,N)[:N]
    return index
    


# In[237]:


with open('matias\Technion\seismic waves\project\data_0.txt') as f:
    data0 = f.readlines()
n0 = np.size(data0)
data = []
for i in range(n0):
    new = np.array(list(eval(data0[i])))
    data.append(new)
data0 = data # now the data is a list of arrays containing integers 
N = 3900
"""
fig, ax = plt.subplots(n0, 1)
[ax[i].plot(np.arange(N), data0[i]) for i in range(n0)]
plt.show()
"""
VARIANCE = False
MEAN_VARIANCE = False
DM_CHOOSE = True
KNN = False

wanted_amount = 6

if VARIANCE:
    best_idx = varience_choose(data0, wanted_amount)
if MEAN_VARIANCE:
    best_idx = mean_varience_choose(data0, wanted_amount)
if DM_CHOOSE:
    best_idx = dm_choose(data0, wanted_amount)
if KNN:
    best_idx = knn_choose(data0, wanted_amount)
    
best_idx_size = best_idx.size  
final_data0 = []
for i in range(best_idx_size):
    final_data0.append(data0[i])
"""
figure, ax1 = plt.subplots(best_idx_size, 1)
[ax1[i].plot(np.arange(N), final_data0[i]) for i in range(best_idx_size)]
plt.show()
"""


print(best_idx)


# In[306]:


def create_ref(im):# in rgb
    plt.figure()
    plt.imshow(book_img)
    plt.title('Original image', fontsize=20)
    corners = plt.ginput(4, 12000) #4 corners for a rectangle
    
    H,W =(500,500)
    rect_p = np.array([[0, 0], [W, 0], [W, H], [0, H]]) # high-left, high-right,low-right,low-left
    Hmat = computeH(rect_p,corners)
    ref_image = warpH(im,Hmat,(W,H))
    return ref_image


# In[307]:


def im2im(ref_img, scene, new_img, d):
    # ref_image - is the referance for the original rectangle
    # scene - is the image that contains the original picture and the one that we want to change
    # new_image - is the new rectangle we want to plant instead of the original 
    (p1,p2) = getPoints_SIFT(ref_img, scene, d)
    Href2scene = computeH(p1,p2) 
    H,W,_ = ref_img.shape

    new_img = cv2.resize(new_img, (W, H)) # making the new image be in the same size as the ref
    new_img2scene = warpH(new_img,Href2scene,scene.shape[:2])
    
    # creating a mask 
    mask = np.zeros(np.shape(new_img2scene))
    mask[new_img2scene != 0] = 1 # holds the places where the new rectangle lies  
    neg_mask = np.ones(np.shape(mask))-mask # holds the places where the new rectangle doesnt lie
    
    new_scene = mask*new_img2scene + negmask*scene 

    return np.uint8(new_scene)

