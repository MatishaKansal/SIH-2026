from .model_utils import balanced_weights
def train(X,y,Xv,yv):
    from xgboost import XGBClassifier
    w=balanced_weights(y); sample_weight=[w[int(v)] for v in y]
    m=XGBClassifier(n_estimators=400,learning_rate=.03,max_depth=6,subsample=.8,colsample_bytree=.8,random_state=1337,eval_metric='auc',early_stopping_rounds=20)
    m.fit(X,y,sample_weight=sample_weight,eval_set=[(Xv,yv)],verbose=False); return m
