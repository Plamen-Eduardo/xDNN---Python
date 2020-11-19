"""
Please cite:
    
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.
"""

import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.special import softmax



def xDNN(Input,Mode):
    if Mode == 'Learning':
        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = max(Labels)
        Prototypes = PrototypesIdentification(Images,Features,Labels,CN)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in range(0,CN+1):
           MemberLabels[i]=Input['Labels'][Input['Labels']==i] 
        Output['xDNNParms']['CurrentNumberofClass']=CN+1
        Output['xDNNParms']['OriginalNumberofClass']=CN+1
        Output['xDNNParms']['MemberLabels']=MemberLabels
        return Output
    
    elif Mode == 'Validation':
        Params=Input['xDNNParms']
        datates=Input['Features']
        Test_Results = DecisionMaking(Params,datates)
        EstimatedLabels = Test_Results['EstimatedLabels'] 
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'],Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
        return Output
    
def PrototypesIdentification(Image,GlobalFeature,LABEL,CL):
    data = {}
    image = {}
    label = {}
    Prototypes = {}
    for i in range(0,CL+1):
        seq = np.argwhere(LABEL==i)
        data[i]=GlobalFeature[seq,]
        image[i] = {}
        for j in range(0, len(seq)):
            image[i][j] = Image[seq[j][0]]
        label[i] = np.ones((len(seq),1))*i
    for i in range(0, CL+1):
        Prototypes[i] = xDNNclassifier(data[i],image[i])
    return Prototypes
        

def xDNNclassifier(Data,Image):
    L, N, W = np.shape(Data)
    radius =1 - math.cos(math.pi/6)
    data = Data.copy()
    Centre = data[0,]
    Center_power = np.power(Centre,2)
    X = np.sum(Center_power)
    Support =np.array([1])
    Noc = 1
    GMean = Centre.copy()
    Radius = np.array([radius])
    ND = 1
    VisualPrototype = {}
    VisualPrototype[1] = Image[0]
    for i in range(2,L+1):
        GMean = (i-1)/i*GMean+data[i-1,]/i
        CentreDensity=np.sum((Centre-np.kron(np.ones((Noc,1)),GMean))**2,axis=1)
        CDmax=max(CentreDensity)
        CDmin=min(CentreDensity)
        DataDensity=np.sum((data[i-1,] - GMean) ** 2)
        if i == 2:
            distance = cdist(data[i-1,].reshape(1,-1),Centre.reshape(1,-1),'euclidean')[0]
        else:
            distance = cdist(data[i-1,].reshape(1,-1),Centre,'euclidean')[0]
        value,position= distance.max(0),distance.argmax(0)
        value=value**2
        
        if DataDensity > CDmax or DataDensity < CDmin or value > 2*Radius[position]:
            Centre=np.vstack((Centre,data[i-1,]))
            Noc=Noc+1
            VisualPrototype[Noc]=Image[i-1]
            X=np.vstack((X,ND))
            Support=np.vstack((Support, 1))
            Radius=np.vstack((Radius, radius))
        else:
            Centre[position,] = Centre[position,]*(Support[position]/Support[position]+1)+data[i-1]/(Support[position]+1)
            Support[position]=Support[position]+1
            Radius[position]=0.5*Radius[position]+0.5*(X[position,]-sum(Centre[position,]**2))/2  
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centre
    dic['Support'] =  Support
    dic['Radius'] =  Radius
    dic['GMean'] =  GMean
    dic['Prototype'] = VisualPrototype
    dic['L'] =  L
    dic['X'] =  X
    return dic  
 
    

def DecisionMaking(Params,datates):
    PARAM=Params['Parameters']
    CurrentNC=Params['CurrentNumberofClass']
    LAB=Params['MemberLabels']
    VV = 1
    LTes=np.shape(datates)[0]
    EstimatedLabels = np.zeros((LTes))
    Scores=np.zeros((LTes,CurrentNC))
    for i in range(1,LTes + 1):
        data = datates[i-1,]
        R=np.zeros((VV,CurrentNC))
        Value=np.zeros((CurrentNC,1))
        for k in range(0,CurrentNC):
            distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'minkowski',6))[0]
            #distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean'))[0]
            Value[k]=distance[0]
        Value = softmax(-1*Value**2).T
        Scores[i-1,] = Value
        Value = Value[0]
        Value_new = np.sort(Value)[::-1]
        indx = np.argsort(Value)[::-1]
        EstimatedLabels[i-1]=indx[0]
    LABEL1=np.zeros((CurrentNC,1))
    
    
    for i in range(0,CurrentNC): 
        LABEL1[i] = np.unique(LAB[i])

    EstimatedLabels = EstimatedLabels.astype(int)
    EstimatedLabels = LABEL1[EstimatedLabels]   
    dic = {}
    dic['EstimatedLabels'] = EstimatedLabels
    dic['Scores'] = Scores
    return dic
         
###############################################################################

