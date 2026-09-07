import tempfile
from .model_utils import balanced_weights
def train(X,y,Xv,yv):
    from catboost import CatBoostClassifier
    m=CatBoostClassifier(iterations=400,learning_rate=.03,depth=6,random_seed=1337,verbose=False,class_weights=balanced_weights(y),eval_metric='AUC',od_type='Iter',od_wait=20,allow_writing_files=False,train_dir=tempfile.gettempdir())
    m.fit(X,y,eval_set=(Xv,yv),early_stopping_rounds=20,verbose=False); return m


