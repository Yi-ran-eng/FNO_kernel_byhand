import pandas as pd
from scipy.stats import norm
import numpy as np
# df=pd.read_excel("D:/house_price_dataset.xlsx")
# featureshead=['房间数量(a)','楼层数(b)','面积_平米(c)','房龄_年(d)']
# datas=featureshead.copy()
# datas.append('房价')
# x=df[featureshead].to_numpy()
# y=df['房价_万元'].to_numpy()
class normalize_centralize:
    sortedarray=[]
    def __init__(self,*args,**kw):
        if (not kw) and args:
            x=args[0]
            self.newx=np.zeros(x.shape)
        if (not args) and kw:
            x=kw.get('x')
            self.newx=np.zeros(x.shape)

    def backcentral(self,x:np.ndarray):
        self.x=x
        for feat in range(x.shape[1]):
            allx_i=x[:,feat]
            maxx,minx=allx_i.max(),allx_i.min()
            medial=(maxx+minx)/2
            crssed=maxx-minx
            self.newx[:,feat]=(allx_i-medial)/crssed
        return self.newx
    def backzero_one(self,x:np.ndarray):
        self.newx=np.zeros(x.shape)
        for feat in range(x.shape[1]):
            allx_f=x[:,feat]
            print(allx_f,'x_feat')
            maxx=allx_f.max()
            print(maxx,'max')
            self.newx[:,feat]=allx_f/maxx
        return self.newx
    def backBox_Nor(self,x:np.ndarray):
        self.x=x
        filt=x > 0
        assert filt.all(),'输入数据必须全正'
        samples=x.shape[0]
        self.sa=samples
        features=x.shape[1]
        p=[]
        for feat in range(x.shape[1]):
            xpiece=x[:,feat].squeeze()
            xsorted=np.sort(xpiece)
            self.sortedarray.append(xsorted)
            for s in range(1,samples+1):
                p.append(s/(samples+1))
            newps=np.array([
                norm.ppf(x) for x in p
            ])
            insetdic={
                xsorted[k]:newps[k] for k in range(samples)
            }#正态分布映射关系
            setattr(self,f'featuredic_{feat-1}',insetdic)
        m=0
        #得到新的数组
        while m < features:
            self.newx[:,m]=np.array(
                [getattr(self,f'featuredic_{m}')[x] for x in x[:,m]]
                )
            m+=1
        return self.newx
    def addnewx_Nor(self,x,featnum:int):
        '''
        featnum给出了是第几个feature
        '''
        if not normalize_centralize.sortedarray:
            raise ValueError('需要先运行backBox_Nor函数以获取已排序的原始数组列表')
        gettarget=normalize_centralize.sortedarray[featnum]
        idx=np.searchsorted(gettarget,x)
        if 1 <= idx <= self.sa-2:
            xraw,xnxt=gettarget[idx],gettarget[idx+1]
            yraw=getattr(self,f'featuredic_{featnum}')[xraw]
            ynxt=getattr(self,f'featuredic_{featnum}')[xnxt]
            ynew=yraw+(x-xraw)/(xnxt-xraw)*(ynxt-yraw)
        elif idx == 0:
            ynew=norm.ppf(1e-4)
        else:
            ynew=norm.ppf(1-1e-4)
        return ynew