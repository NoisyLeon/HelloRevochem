import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numba
import pandas as pd

def read_raw_data(infname):
    # # # data        = list(csv.reader(open('./Raw_Data/CP345979_img01.csv')))
    data        = list(csv.reader(open(infname)))
    data[0][0]  = '0.0'
    data        = np.asarray(data, dtype=float)
    xarr        = data[0, 1:]
    yarr        = data[1:, 0]
    x, y        = np.meshgrid(xarr, yarr)
    data        = data[1:, 1:]
    return data, xarr, yarr

@numba.jit(numba.int32[:](numba.float32[:], numba.float32[:], numba.float32[:]))
def get_valid_index(t, dt, tarr):
    index       = np.zeros(t.size, dtype=int)
    for i in range(t.size):
        # tmin    = t[i] - dt[i]/60.
        # tmax    = t[i] + dt[i]/60.
        
        # if abs(t[i] - 50.79)<0.05:
        #     continue
        # index   += abs(tarr - t[i])<0.05
        index[i]    = np.where(abs(tarr - t[i])<0.05)[0]
    return index

data1, x, y     = read_raw_data('./Raw_Data/CP345979_img01.csv')
data2, x, y     = read_raw_data('./Raw_Data/CP345980_img01.csv')
data, x, y      = read_raw_data('./Raw_Data/CPMIX3.csv')

blob1   = (pd.read_excel('./Blob_Table/CP345979_New_Blob_Table.xls')).as_matrix()
# cpname  = 
t1      = np.array(blob1[:60, 5], dtype=np.float32)
dt1     = np.array(blob1[:60, 6], dtype=np.float32)

blob2   = (pd.read_excel('./Blob_Table/CP345980_New_Blob_Table.xls')).as_matrix()
t2      = np.array(blob2[:60, 5], dtype=np.float32)
dt2     = np.array(blob2[:60, 6], dtype=np.float32)

blob3   = (pd.read_excel('./Blob_Table/CPMIX3_New_Blob_Table.xls')).as_matrix()
t3      = np.array(blob3[:60, 5], dtype=np.float32)
dt3     = np.array(blob3[:60, 6], dtype=np.float32)


ind1    = get_valid_index(t1, dt1, x)
ind2    = get_valid_index(t2, dt2, x)
ind3    = get_valid_index(t3, dt3, x)

data1   = data1[:, ind1]
data2   = data2[:, ind2]
data    = data[:, ind3]
# 
# plt.pcolormesh(x, y, data, norm=colors.LogNorm(vmin=1., vmax=data.max()))
# plt.show()

N=1000
misfitarr   = np.zeros(N)
betaarr     = np.zeros(N)
for i in range(N):
    beta1   = float(i)/float(N)
    beta2   = 1. - beta1
    # misfit  = np.sqrt(((beta1*ph1/vol1 + beta2*ph2/vol2 - ph3/vol3)**2).sum()/ph1.size)
    misfit  = np.sqrt(((beta1*data1 + beta2*data2 - data)**2).sum()/data1.size)
    # misfit  = np.sqrt(((beta1*vol1 + beta2*vol2 - vol3)**2).sum()/ph1.size)
    # print (beta1, misfit)
    misfitarr[i]= misfit
    betaarr[i]  = beta1

indmin  = misfitarr.argmin()
print (betaarr[indmin])