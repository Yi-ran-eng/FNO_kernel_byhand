# FNO_kernel_byhand

__this is a FNO kernel created by hand, it cannot be used in machine learning or deeplearning Cuz there is only a kernel. But it shows how fno kernel is built and embedded in normal nerual networks__

I hope my works will help others understand this process

* in initia file,i created several methods for initializing random datas, but that mainly for weights and bias,not x ,or say, input features
* in FNO file,i wrote two classes ,one for FNO Initializing and another is for linear layer initializing, i chose setter_property approaches in Python to help me update these parameters
* in args file,to be more normal, i used argsparse to save datas along with dict_saving ,this dic_saving approach was used to compensated for argsparse's trait about no support for np.ndarray
* TRAIN and predict file are there to show how to built a training process and predict process by combining these ways

i don't add activating factor to this data training networks, and it is the reason why this network cannot be used in model training ,and remember it's normal to find this network can't predict well in many cases.

to use it well ,add activating factor ,modes,dropout layers and maybe more nerual layers into this code, essetially, transfer this model onto torch or tensorflow.

__the file predicted_opted is uploaded later, for such features that contain different number ranges with significant differences , it's strongly recommended to use normalization before training, in this code it's normalize_centralize in Initialize.py ,in this way ,the accuracy of prediction is much higher than before__
