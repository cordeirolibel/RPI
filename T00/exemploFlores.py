
# coding: utf-8

# In[3]:

from skimage import io, color, exposure
import numpy as np
import os
from matplotlib import pyplot as plt
get_ipython().magic('pylab inline')

d0 = []
d1 = []
d2 = []

for img in os.listdir('flores/'):
    im=io.imread('flores//' + img)          
    print('Processing ' + img)
    
    im = color.rgb2hsv(im);  
   
    #descritor simples - soma de histograma
    aux1 = [x/64 for x in im[:,:,0]]
    aux2 = [x/16 for x in im[:,:,1]]
    aux3 = [x/4 for x in im[:,:,2]]
    idx = np.add(aux1,np.add(aux2,aux3))
    d0.append(np.sum(idx))
    
    #média do valor
    idx1 = np.mean(im[:,:,2])
    d1.append(idx1)    
    
    #média de um ajuste do valor
    aux = [np.square(x*2) for x in im[:,:,2]]
    idx2 = np.mean(aux);
    d2.append(idx2)
  
#plots dos resultados
plt.figure()
plt.plot(d1[0:5], d0[0:5], 'rs')
plt.plot(d1[5:10], d0[5:10], 'k*')
plt.plot(d1[11:15], d0[11:15], 'yo')
plt.plot(d1[16:20], d0[16:20], 'bo')

plt.figure()
plt.plot(d2[0:5], d0[0:5], 'rs')
plt.plot(d2[5:10], d0[5:10], 'k*')
plt.plot(d2[11:15], d0[11:15], 'yo')
plt.plot(d2[16:20], d0[16:20], 'bo')


# In[ ]:



