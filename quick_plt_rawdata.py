import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# data        = list(csv.reader(open('./Raw_Data/CP345979_img01.csv')))
data        = list(csv.reader(open('./Raw_Data/CP345980_img01.csv')))

data[0][0]  = '0.0'
data        = np.asarray(data, dtype=float)
xarr        = data[0, 1:]
yarr        = data[1:, 0]
x, y        = np.meshgrid(xarr, yarr)
data        = data[1:, 1:]

plt.pcolormesh(x, y, data, norm=colors.LogNorm(vmin=1., vmax=data.max()))
plt.show()
