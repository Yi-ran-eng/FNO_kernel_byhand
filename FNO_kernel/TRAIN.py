from Fnopre import FNOLayerinset,Linerlink
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC,abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argsconfig import main
class MSELOSSvision:
    '''
    param y:正确值
    param y_hat:预测值
    param m:样本数
    '''
    @abstractmethod
    def __init__(self,*args,**kw):
        pass
    def Jloss(self,y_hat,y):
        res=(y_hat-y)**2
        self.L=res.sum(axis=0)/(2*len(y.squeeze()))
        return self.L
    def highmap(self,ws,bs,level,**kw):
        # assert ws.shape == bs.shape,'权重和偏置维度不匹配'
        if len(ws.shape) == 2:
            #削减成一维的
            w=np.squeeze(ws)
            b=np.squeeze(bs)
        # print(w.shape,'w')
        W,B=np.meshgrid(w,b)
        self.W=W
        self.B=B
        canzhaoL=kw.get('Lin')
        wbs=kw.get('wbs')
        fig,ax=plt.subplots()
        if canzhaoL is None:
            assert self.L.T.shape == W.shape,f'损失函数形状有问题，目前损失函数形状是{self.L.shape}'
            #捕捉kw
            ax.contour(self.W,self.B,self.L.T,levels=level,cmap='viridis')
            self.golim=self.L.T.copy()
            self.xll,self.yll=ax.get_xlim(),ax.get_ylim()
        else:
            ax.contour(self.W,self.B,canzhaoL,levels=level,cmap='viridis')
            if hasattr(self,'xll') and hasattr(self,'yll'):
                ax.set_xlim(self.xll)
                ax.set_ylim(self.yll)
        ax.set_xlabel('W')
        ax.set_ylabel('b')
        if canzhaoL is None:
            return self.golim
    @abstractmethod
    def gradown(self,*args,**kwt):
        pass
    def _3dmap(self):
        fig=plt.figure()
        axx=fig.add_subplot(111,projection='3d')
        surf=axx.plot_surface(self.W,self.B,self.L.T,cmap='viridis',edgecolor='none',alpha=0.8)
        fig.colorbar(surf,shrink=0.5,aspect=10,label='loss_visiable')
        # axx.set_xlim(-5,5)
        axx.set_xlabel('W')
        axx.set_ylabel('B')
        axx.set_zlabel('Loss')
    def seeloss(self,losses):
        fig,ax=plt.subplots()
        ax.plot(losses)
        ax.set_title(f'最终loss={losses[-1]:.4f}')
        plt.show()
class Model_FNO(MSELOSSvision):
    def __init__(self,x:np.ndarray,y:np.ndarray,twoside=False,maxiter=100,fnonum=10):
        self.features,self.targets=x,y
        self.sample,self.feats=x.shape
        self.iter=maxiter
        self.fnonum=fnonum
        self.two=twoside
    def forward(self,f:FNOLayerinset,**kw):
        f.wrapsin(self.two)
        self.fnowraps=f.F
        vc=kw.get('v_c')
        if vc is None:
            self.link=Linerlink(self.fnowraps,v_cpams=vc)
        else:
            self.link.F=self.fnowraps
        y_hat=self.link.linklay()
        self.hat=y_hat
    def backward(self):
        self.dLdY_hat=2*(self.hat-self.targets.reshape(self.hat.shape))/self.sample
        #这个形状是samples x 1
        self.dLdv=self.fnowraps.T @ self.dLdY_hat#形状：fnonum x 1
        self.dLdc=np.sum(self.dLdY_hat.squeeze())#是一个标量
        self.dLdF=self.dLdY_hat @ self.link.v.T#形状：samples x fnonum
        if self.two:
            self.dLdF_cos=self.dLdF[:,0:self.fnonum]
            self.dLdF_sin=self.dLdF[:,self.fnonum:]
            self.dLdZ=self.dLdF_cos*(-np.sin(self.f.Z))+self.dLdF_sin*np.cos(self.f.Z)
        else:
            self.dLdZ=self.dLdF*(-np.sin(self.f.Z))#samples x fnonum
        self.dLdw=self.features .T @ self.dLdZ#features x fnonum
        self.dLdb=np.sum(self.dLdZ,axis=0)#fnonum,
    def gradown(self,alpha:float,choi_loss=True):
        f=FNOLayerinset(self.features,FNOinm=self.fnonum if self.fnonum is not None 
                        else FNOLayerinset.__init__.__defaults__[0],useGuass=True)
        self.f=f
        if self.fnonum is None:
            self.fnonum=FNOLayerinset.__init__.__defaults__[0]
        losses=[]
        wbpath=[]
        k=0
        self.targets=self.targets.reshape(-1,1)
        while k < self.iter:
            try:
                self.forward(f,v_c=getattr(self,'link'))#这里就是象征性写一下，实则没有用到这个参数
            except AttributeError:
                self.forward(f)
            #取出y_hat
            self.wint,self.bint=f.w0[0,0],f.b0[0]
            wbpath.append((self.wint,self.bint))#记录轨迹
            super().Jloss(self.hat,self.targets)
            losses.append(self.L[0])
            self.backward()
            #更新参数
            if tuple(f.w0.shape) == (int(self.fnonum),int(self.feats)):
                ws=f.w0.T
            else:
                ws=f.w0
            f.w0=(ws-alpha*self.dLdw).T
            f.b0=f.b0-alpha*self.dLdb
            self.link.v=self.link.v-alpha*self.dLdv
            self.link.c=self.link.c-alpha*self.dLdc
            k+=1
        if choi_loss:
            super().seeloss(losses=losses)
        _,invet,rename,self.getparam,_=main(4)
        rename(invet,'w',f.w0)
        rename(invet,'b',f.b0)
        rename(invet,'v',self.link.v)
        rename(invet,'c',self.link.c)
        return self.getparam
    def testprice(self,featx):
        w=self.getparam('w')
        b=self.getparam('b')
        c,v=self.getparam('c'),self.getparam('v')
        self.Z_test=featx @ w.T+b
        if self.two:
            self.F_test=np.hstack((np.cos(self.Z_test),np.sin(self.Z_test)))
        else:
            self.F_test=np.cos(self.Z_test)
        self.Y_hat=self.F_test @ v +c
        return self.Y_hat
