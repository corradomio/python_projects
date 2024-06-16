#!/usr/bin/env python3




from sklearn.linear_model import Perceptron, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor # sklearn.qda
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


import warnings
warnings.filterwarnings("ignore")

def get_reg_models():
   return get_reg_models_except([])

def get_reg_models_except(ignoreList):
  list_regression_models ={}
  for i in range(1,52):
    if i in ignoreList:
        pass
    else:
        reg,name = InitRegressor(i)
        list_regression_models[name] = reg
  return list_regression_models

def InitRegressor(index):
   C = 1.0
   parameters1 = {'kernel': ['linear'], 'gamma': [0.1, 0.01, 1e-3, 1e-4],'C': [1, 10, 100, 1000]}
   parameters2 = {'kernel': ['poly'], 'gamma': [0.1, 0.01, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
  
   if index==1:
       name = 'LinR_CpxFiticpt!Nor' 
       clf=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                normalize=False)       
   if index==2:
       name = 'Ridge_alp0.5Slv-autoTol0.001' 
       clf=Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
   if index==3:
       name = 'Lasso_alp0.1It1000Sel-cyclicTol0.0001' 
       clf=Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
       #clf=ElasticNet()
   if index==4:       
     name = 'KN' 
     clf=KNeighborsRegressor()
   if index==5:  
     #clf=SVR(kernel='linear', C=C, tol=1e-3, verbose=False,cache_size=7000) 
     name = 'SVR_PolyCache7000' 
     clf=SVR(kernel='poly',cache_size=7000)          
   if index==6:
     name = 'SVR_PolyD3Tol1e-4!Verbose' 
     clf =SVR(kernel='poly', degree=3, C=C, tol=1e-4, verbose=False)
   if index==7:
     name = 'SVR_RbfTol1e-4!Verbose' 
     clf =SVR(kernel='rbf', C=C, tol=1e-4, verbose=False)
   if index==8:
     name = 'GBoost_Est100Dep11Samp1.0' 
     clf = GradientBoostingRegressor(n_estimators=100, max_depth=11, subsample=1.0)    
   if index==9:
     name = 'GBoost_Est100Dep11' 
     clf = RandomForestRegressor(max_depth=11, n_estimators=100)    
   if index==10:
     name = 'DT_Samsplit2Rndstate0' 
     clf = DecisionTreeRegressor(max_depth=None, min_samples_split=2,random_state=0)     
   if index==11:
     name = 'ExtraT_Est100Rndstate0' 
     clf = ExtraTreesRegressor(n_estimators=100,random_state=0)     
   if index==12:
     name = 'DT_Dep5' 
     clf = DecisionTreeRegressor(max_depth=5)   
   if index==13:
     name = 'RF_Dep5Est500Fet1' 
     clf = RandomForestRegressor(max_depth=5, n_estimators=500, max_features=1)     
   if index==14:
     name = 'GBoost_Dep11Est500' 
     clf = GradientBoostingRegressor(n_estimators=500, max_depth=11)     
   if index==15:
     name = 'SGD_Pen11' 
     clf = linear_model.SGDRegressor(penalty='l1')
   if index==16:
     name = 'SGD_Pen11Loss-huber' 
     clf = linear_model.SGDRegressor(loss='huber',penalty='l1')
   if index==17:
     name = 'SGD_Pen11Loss-eps' 
     clf = linear_model.SGDRegressor(loss='epsilon_insensitive',penalty='l1')
   if index==18:
     name = 'SGD_Pen12' 
     clf = linear_model.SGDRegressor(penalty='l2')
   if index==19:  
     name = 'KN_N5' 
     clf=KNeighborsRegressor(n_neighbors=5)
   if index==20:
     name = 'GBoost_Dep3Est500' 
     clf = GradientBoostingRegressor(n_estimators=500, max_depth=3)   
   if index==21:
     name = 'GBoost_Dep5Est500' 
     clf = GradientBoostingRegressor(n_estimators=500, max_depth=5)   
   if index==22:
     name = 'RF_Dep11Est500' 
     clf = RandomForestRegressor(max_depth=11, n_estimators=500)  
   if index==23:
     name = 'RF_Dep13Est700' 
     clf = RandomForestRegressor(max_depth=13, n_estimators=700)   
   if index==24:
     name = 'GBoost_Dep11Est100' 
     clf = GradientBoostingRegressor(n_estimators=100, max_depth=11)   # 1000 for 23,24,25,26
   if index==25:
     name = 'GBoost_Dep12Est100' 
     clf = GradientBoostingRegressor(n_estimators=100, max_depth=12)   
   if index==26:
     name = 'GBoost_Dep13Est100'      
     clf = GradientBoostingRegressor(n_estimators=100, max_depth=13)   
   if index==27:
     name = 'GBoost_Dep14Est100' 
     clf = GradientBoostingRegressor(n_estimators=100, max_depth=14)   
   if index==28:
     name = 'SGD_Loss-sqPen11' 
     clf = linear_model.SGDRegressor(loss='squared_loss', penalty='l1') 
   if index==29:
     name = 'SGD_Loss-sqPen12' 
     clf = linear_model.SGDRegressor(loss='squared_loss', penalty='l2' )     
   if index==30:
     name = 'SGD_Loss-epPen11' 
     clf = linear_model.SGDRegressor(loss='epsilon_insensitive',penalty='l1') 
   if index==31:
     name = 'SGD_Loss-epPen12' 
     clf = linear_model.SGDRegressor(loss='epsilon_insensitive',penalty='l2') 
   if index==32:
     name = 'GSCV_SVR-lin!verboseLin' 
     clf = GridSearchCV(SVR(kernel='linear',verbose=False),parameters1,verbose=False)
   if index==33:
     name = 'RF_Dep5Est50' 
     clf = RandomForestRegressor(max_depth=5, n_estimators=50)    
   if index==34:
     name = 'GSCV_SVR-poly!verboseLinPoly' 
     clf = GridSearchCV(SVR(kernel='poly',verbose=False),parameters2,verbose=False)
   if index==35:
     name = 'GBoost_Dep5Est25Sam1.0' 
     clf = GradientBoostingRegressor(n_estimators=25, max_depth=5, subsample=1.0)  # This
   if index==36:
     name = 'GBoost_Dep5Est30Sam1.0' 
     clf = GradientBoostingRegressor(n_estimators=30, max_depth=5, subsample=1.0) 
   if index==37:
     name = 'GBoost_Dep5Est40Sam1.0' 
     clf = GradientBoostingRegressor(n_estimators=40, max_depth=5, subsample=1.0) 
   if index==38:
     name = 'RF_Dep7Est80' 
     clf = RandomForestRegressor(max_depth=7, n_estimators=80)
   if index==39:
     name = 'RF_Dep7Est100' 
     clf = RandomForestRegressor(max_depth=7, n_estimators=100)
   if index==40:
     name = 'RF_Dep7Est120' 
     clf = RandomForestRegressor(max_depth=7, n_estimators=120)
   if index==41:
     name = 'RF_Dep9Est150' 
     clf = RandomForestRegressor(max_depth=9, n_estimators=150) # This
   if index==42:
     name = 'RF_Dep5Est10' 
     clf = RandomForestRegressor(max_depth=5, n_estimators=10)
   if index==43:
     name = 'RF_Dep5Est15' 
     clf = RandomForestRegressor(max_depth=5, n_estimators=15)
   if index==44:
     name = 'RF_Dep9Est10' 
     clf = RandomForestRegressor(max_depth=9, n_estimators=10)
   if index==45:
     name = 'RF_Dep11Est10' 
     clf = RandomForestRegressor(max_depth=11, n_estimators=10) # This
   if index==46:
     name = 'RF_Dep11Est15' 
     clf = RandomForestRegressor(max_depth=11, n_estimators=15) # This
   if index==47:
     name = 'DT_Dep7' 
     clf = DecisionTreeRegressor(max_depth=7)   
   if index==48:
     name = 'DT_Dep10' 
     clf = DecisionTreeRegressor(max_depth=10)   
   if index==49:
     name = 'DT_Dep15' 
     clf = DecisionTreeRegressor(max_depth=15)   
   if index==50:
     name = 'DT_Dep20' 
     clf = DecisionTreeRegressor(max_depth=20)   
   if index==51:
     name = 'MLP_Act-idntSol-obfgsItr5000Lr-cons' 
     clf=MLPRegressor(activation = 'identity',solver = 'lbfgs',max_iter = 5000,learning_rate = 'constant')
   if index==52:
     name = 'RF' 
     clf=RandomForestRegressor()

   return clf, name





