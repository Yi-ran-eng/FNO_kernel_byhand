'''
this code is intended to show how single FNO kernel works 
Cuz I only complemented one layer without activating funtion,it doesn't predict well

Later we know one integral FNO layer includes the basic bp layer(fno factor layer add basic bp layer)
then it reserves some traits about bp networks
'''
import pandas as pd
from TRAIN import Model_FNO
from Intialdots import normalize_centralize
df=pd.read_excel("D:/house_price_dataset.xlsx")
featureshead=['房间数量(a)','楼层数(b)','面积_平米(c)','房龄_年(d)']
datas=featureshead.copy()
datas.append('房价')
x=df[featureshead].to_numpy()
y=df['房价_万元'].to_numpy()

mor=normalize_centralize()
xpacked=mor.backzero_one(x)
f=Model_FNO(xpacked,y,twoside=True,maxiter=10000,fnonum=100)
f.gradown(alpha=0.0001)
print(x[0:4,:])
print(f.testprice(x[0:9,:],featransfer=True))