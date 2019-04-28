import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

blob1   = (pd.read_excel('./Blob_Table/CP345979_New_Blob_Table.xls')).as_matrix()
t1      = np.array(blob1[:60, 5], dtype=float)
ind     = t1.argsort()
t1      = t1[ind]
ph1     = np.array(blob1[ind, 7], dtype=float)
vol1    = np.array(blob1[ind, 8], dtype=float)

blob2   = (pd.read_excel('./Blob_Table/CP345980_New_Blob_Table.xls')).as_matrix()
t2      = np.array(blob2[:60, 5], dtype=float)
ind     = t2.argsort()
t2      = t2[ind]
ph2     = np.array(blob2[ind, 7], dtype=float)
vol2    = np.array(blob2[ind, 8], dtype=float)

blob3   = (pd.read_excel('./Blob_Table/CPMIX3_New_Blob_Table.xls')).as_matrix()
t3      = np.array(blob3[:60, 5], dtype=float)
ind     = t3.argsort()
t3      = t3[ind]
ph3     = np.array(blob3[ind, 7], dtype=float)
vol3    = np.array(blob3[ind, 8], dtype=float)

# plt.plot(t1, ph1, 'o')
# # plt.yscale('log', linthreshy=0.01)
# plt.plot(t2, ph2, '^')
# plt.plot(t3, ph3, 'v')
# plt.show()
# 
# blob1   = (pd.read_excel('./Blob_Table/CP345979_New_Blob_Table.xls')).as_matrix()
# t1      = np.array(blob1[:, 5], dtype=float)
# ind     = t1.argsort()
# t1      = t1[ind]
# ph1     = np.array(blob1[ind, 7], dtype=float)
# 
# blob2   = (pd.read_excel('./Blob_Table/CP345980_New_Blob_Table.xls')).as_matrix()
# t2      = np.array(blob2[:, 5], dtype=float)
# ind     = t2.argsort()
# t2      = t2[ind]
# ph2     = np.array(blob2[ind, 7], dtype=float)
# 
# blob3   = (pd.read_excel('./Blob_Table/CPMIX1_New_Blob_Table.xls')).as_matrix()
# t3      = np.array(blob3[:, 5], dtype=float)
# ind     = t3.argsort()
# t3      = t3[ind]
# ph3     = np.array(blob3[ind, 7], dtype=float)
# 
# plt.plot(t1, ph1)
# # plt.yscale('log', linthreshy=0.01)
# plt.plot(t2, ph2)
# plt.plot(t3, ph3)
# plt.show()

N=1000
misfitarr   = np.zeros(N)
betaarr     = np.zeros(N)
for i in range(N):
    beta1   = float(i)/float(N)
    beta2   = 1 - beta1
    # misfit  = np.sqrt(((beta1*ph1/vol1 + beta2*ph2/vol2 - ph3/vol3)**2).sum()/ph1.size)
    misfit  = np.sqrt(((beta1*ph1 + beta2*ph2 - ph3)**2).sum()/ph1.size)
    # misfit  = np.sqrt(((beta1*vol1 + beta2*vol2 - vol3)**2).sum()/ph1.size)
    # print (beta1, misfit)
    misfitarr[i]= misfit
    betaarr[i]  = beta1

indmin  = misfitarr.argmin()
print (betaarr[indmin])

