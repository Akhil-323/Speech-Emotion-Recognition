'''
Created on 2024-10-14

@author: loson
'''

import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from pre_processing import split_data
from sklearn.mixture import GaussianMixture


def model_gmm():
    X_train, X_test, y_train, y_test = split_data()
    classes = len(np.unique(y_train))
    
    train_result_dict={}
    test_result_dict={}
    
    # Try GMMs using different types of covariance.
    models = {
        co_ty: GaussianMixture(
            n_components=classes, covariance_type=co_ty, max_iter=1000
        )
        for co_ty in ["spherical", "diag", "tied", "full"]
        }
    
    for index, (name, model) in enumerate(models.items()):
        
        # Train the other parameters using the EM algorithm.
        model.fit(X_train)
    
        y_train_pred = model.predict(X_train)
        train_result_dict[name]= y_train_pred
        
        y_test_pred = model.predict(X_test)
        test_result_dict[name]=y_test_pred
                
    return train_result_dict,test_result_dict,y_train, y_test
 
def evaluation():
    train_result_dict,test_result_dict,y_train, y_test=model_gmm()
    
    for name,y_train_pred in train_result_dict.items():
        acc_score_train=accuracy_score(y_true=y_train,y_pred=y_train_pred)
        print("covariance_type: "+ name+"-- training accuracy_score:"+ str(acc_score_train))
        
        print(classification_report(y_train,y_train_pred)) 
        print(confusion_matrix(y_train, y_train_pred))
        
    for name,y_test_pred in test_result_dict.items():
        acc_score_test=accuracy_score(y_true=y_test,y_pred=y_test_pred)
        print("covariance_type: "+ name+"-- testing accuracy_score:"+ str(acc_score_test))
        
        print(classification_report(y_test,y_test_pred))
        print(confusion_matrix(y_test, y_test_pred))
    return       

test=evaluation()


 
