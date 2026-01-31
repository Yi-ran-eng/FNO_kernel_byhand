import argparse
import math
#创建解析器
def tool(name):
    parse=argparse.ArgumentParser(description=name)
    return parse
def main(parms):
    parse=tool(parms)
    cc=parse.add_subparsers(dest='command',help='_saved')
    invet=cc.add_parser('pams')
    paramstore={}

    def rename(invet,name:str,value=None):
        invet.add_argument(f'--{name}',dest=f'_{name}',default='')
        if value is not None:
            paramstore[f'_{name}']=value
    def getparams(name):
        return paramstore.get(f'_{name}')
    def setparams(name,value):
        paramstore[f'_{name}']=value
    invet.set_defaults(func=lambda a:print('没有任何参数传入'))
    return parse,invet,rename,getparams,setparams