import rasterio
from rasterio.plot import show

fp = r"D:\Dropbox\2_Khalifa\Progetto PLACES\data\uaepopulation\are_ppp_2020.tif"

dataset = rasterio.open(fp)
# show(img)

print(dataset.count)


from osgeo import gdal
import matplotlib.pyplot as plt

dataset = gdal.Open(fp, gdal.GA_ReadOnly)
# Note GetRasterBand() takes band no. starting from 1 not 0
band = dataset.GetRasterBand(1)
arr = band.ReadAsArray()
plt.imshow(arr)
plt.show()