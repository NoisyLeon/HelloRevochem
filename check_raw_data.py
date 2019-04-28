import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def read_raw_data(infname):
    # # # data        = list(csv.reader(open('./Raw_Data/CP345979_img01.csv')))
    data        = list(csv.reader(open(infname)))
    data[0][0]  = '0.0'
    data        = np.asarray(data, dtype=float)
    xarr        = data[0, 1:]
    yarr        = data[1:, 0]
    x, y        = np.meshgrid(xarr, yarr)
    data        = data[1:, 1:]
    return data, x, y 

data1, x, y     = read_raw_data('./Raw_Data/CP345979_img01.csv')
data2, x, y     = read_raw_data('./Raw_Data/CP345980_img01.csv')
data, x, y      = read_raw_data('./Raw_Data/CPMIX2.csv')

data1   = data1[x>30.]
data2   = data2[x>30.]
data    = data[x>30.]

# 
# plt.pcolormesh(x, y, data, norm=colors.LogNorm(vmin=1., vmax=data.max()))
# plt.show()

N=100
misfitarr   = np.zeros(N)
betaarr     = np.zeros(N)
for i in range(N):
    beta1   = float(i)/float(N)
    beta2   = 1 - beta1
    # misfit  = np.sqrt(((beta1*ph1/vol1 + beta2*ph2/vol2 - ph3/vol3)**2).sum()/ph1.size)
    misfit  = np.sqrt(((beta1*data1 + beta2*data2 - data)**2).sum()/data1.size)
    # misfit  = np.sqrt(((beta1*vol1 + beta2*vol2 - vol3)**2).sum()/ph1.size)
    # print (beta1, misfit)
    misfitarr[i]= misfit
    betaarr[i]  = beta1

indmin  = misfitarr.argmin()
print (betaarr[indmin])