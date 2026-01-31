import numpy as np
from Intialdots import normalize_centralize

class FNOLayerinset:
    def __init__(self,x:np.ndarray,FNOinm=10,useGuass=False):
        '''
        singlize:使用单一线性网络
        '''
        assert np.ndim(x) == 2,'输入特征的维度必须是2，每行是一组features'
        self.input=x
        self.samples,self.features=x.shape[0],x.shape[1]
        # self._w0=np.ones((FNOinm,self.features))
        if not useGuass:
            n=self.features*FNOinm
            cov=np.eye(n)
            means=np.zeros(n)
            w0=np.random.multivariate_normal(means,cov,1)
            self._w0=w0.reshape(FNOinm,self.features)
        else:
            self._w0=self._2Dguass_kernel(FNOinm,self.features)
        self._b0=np.zeros(FNOinm)
    @property
    def w0(self):
        return self._w0
    @w0.setter
    def w0(self,updatedw:np.ndarray)->np.ndarray:
        if updatedw is None:
            raise ValueError('更新值不能为空')
        self._w0=updatedw
        return self._w0
    @property
    def b0(self):
        return self._b0
    @b0.setter
    def b0(self,updatedb:np.ndarray)->np.ndarray:
        if updatedb is None:
            raise ValueError('更新值不能为空')
        self._b0=updatedb
        return self._b0
    def LIner(self):
        '''
        输出形状：samples x fnonum
        '''
        return self.input @ self.w0.T + self.b0
    @staticmethod
    def _2Dguass_kernel(rows,cols,mux=None,muy=None,sigx=1.0,sigy=1.0)->np.ndarray:
        '''
        如果没有传入mux和muy将使用归一化后的中值
        '''
        print(f'cols={cols},rows={rows}')
        x=np.arange(cols)
        y=np.arange(rows)
        norop=normalize_centralize()
        xnored=norop.backzero_one(x.reshape(-1,1)) if cols >= 2 else x
        ynored=norop.backzero_one(y.reshape(-1,1)) if rows >= 2 else y
        if mux is None:
            mux=float(xnored.squeeze().max()/2)
        if muy is None:
            muy=float(ynored.squeeze().max()/2)
        X,Y=np.meshgrid(xnored,ynored)
        guassion=np.exp(-((X-mux)**2/(2*sigx**2)+(Y-muy)**2/(2*sigy**2)))
        guassion=guassion/np.sum(guassion)
        guassion-=np.mean(guassion)
        return guassion
    def wrapsin(self,twoside=False):
        linans=self.LIner()
        self.Z=linans
        if not twoside:
            self.F=np.cos(linans)
        else:
            self.F=np.hstack((np.cos(linans),np.sin(linans)))
        return self.F
class Linerlink:
    def __init__(self,F:np.ndarray,**kw):
        '''
        F的形状：samples x fnonum
        '''
        self._F=F
        vc=kw.get('v_cpams')
        if vc is None:#意思是初始化
            self._v=FNOLayerinset._2Dguass_kernel(F.shape[1],1,mux=0.5,muy=0.5)
            self._c=1.0
        else:
            self._v=vc[0]
            self._c=vc[1]
    @property
    def F(self):
        return self._F
    @F.setter
    def F(self,newF:np.ndarray):
        self._F=newF
        return self._F
    @property
    def v(self):
        return self._v
    @v.setter
    def v(self,newv:np.ndarray):
        self._v=newv
        return self._v
    @property
    def c(self):
        return self._c
    @c.setter
    def c(self,newc:np.ndarray):
        self._c=newc
        return self._c
    def linklay(self):
        return self.F @ self.v + self.c
# x=np.array([[2,3,4],[5,4,3],[4,2,1],[6,5,4]])
# y=np.array([3,5,1,2])
# f=FNOLayerinset(x,useGuass=True)
# m=f.wrapsin(1)
# print(m.shape)