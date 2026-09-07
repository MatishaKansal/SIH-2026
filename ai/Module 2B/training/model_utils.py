import time, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve

def scores(model, X):
    if hasattr(model,"predict_proba"): return model.predict_proba(X)[:,1]
    return model.predict(X)
def eer(y,p):
    fpr,tpr,_=roc_curve(y,p); i=np.nanargmin(np.abs(1-tpr-fpr)); return float((fpr[i]+1-tpr[i])/2)
def metrics(y,p, threshold=.5):
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"eer":eer(y,p),"f1":float(f1_score(y,pred,zero_division=0)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"specificity":float(tn/(tn+fp)) if tn+fp else 0.,"accuracy":float((tn+tp)/(tn+fp+fn+tp)),"far":float(fp/(fp+tn)) if fp+tn else 0.,"frr":float(fn/(fn+tp)) if fn+tp else 0.,"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}
def latency_ms(model,X,n=100):
    sample=X[np.arange(min(n,len(X))) % len(X)]; start=time.perf_counter()
    for x in sample: scores(model,x.reshape(1,-1))
    return (time.perf_counter()-start)*1000/len(sample)
def balanced_weights(y):
    counts=np.bincount(np.asarray(y,dtype=int)); return {i:len(y)/(len(counts)*c) for i,c in enumerate(counts) if c}
