"""
Crop Stress Detection & Irrigation Priority — training script
DATA NOTICE: This package's CSV is synthetic demonstration data, not field measurements.
"""
import os, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = os.environ.get("CROP_STRESS_CSV", "crop_stress_dataset_synthetic_demo.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
df = pd.read_csv(DATA_PATH)

excluded = {"field_id","observation_date","stress_risk","stress_index_0_1",
            "irrigation_priority_score_0_100","irrigation_priority_category","data_origin"}
X = df[[c for c in df.columns if c not in excluded]].copy()
yc, yr, groups = df["stress_risk"], df["stress_index_0_1"], df["field_id"]
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
pre = ColumnTransformer([
    ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                      ("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
tr, te = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=RANDOM_STATE).split(X,yc,groups))
Xtr,Xte=X.iloc[tr],X.iloc[te]
yctr,ycte=yc.iloc[tr],yc.iloc[te]
yrtr,yrte=yr.iloc[tr],yr.iloc[te]
gtr=groups.iloc[tr]

classifiers = {
 "Logistic Regression": LogisticRegression(max_iter=1500,class_weight="balanced"),
 "KNN": KNeighborsClassifier(n_neighbors=7),
 "SVM": SVC(probability=True,class_weight="balanced",random_state=RANDOM_STATE),
 "Decision Tree": DecisionTreeClassifier(class_weight="balanced",random_state=RANDOM_STATE),
 "Random Forest": RandomForestClassifier(n_estimators=120,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1)
}
cls_rows=[]; cls_pipes={}
for name, model in classifiers.items():
    pipe=Pipeline([("preprocess",pre),("model",model)]).fit(Xtr,yctr)
    pred=pipe.predict(Xte)
    p,r,f,_=precision_recall_fscore_support(ycte,pred,average="weighted",zero_division=0)
    cls_rows.append({"model":name,"accuracy":accuracy_score(ycte,pred),
                     "precision_weighted":p,"recall_weighted":r,"f1_weighted":f})
    cls_pipes[name]=pipe
cls_cmp=pd.DataFrame(cls_rows).sort_values("f1_weighted",ascending=False)

regressors={
 "Linear Regression":LinearRegression(),
 "KNN Regressor":KNeighborsRegressor(n_neighbors=7),
 "SVR":SVR(C=5,epsilon=.05),
 "Decision Tree Regressor":DecisionTreeRegressor(random_state=RANDOM_STATE),
 "Random Forest Regressor":RandomForestRegressor(n_estimators=120,random_state=RANDOM_STATE,n_jobs=-1)
}
reg_rows=[]; reg_pipes={}
for name,model in regressors.items():
    pipe=Pipeline([("preprocess",pre),("model",model)]).fit(Xtr,yrtr)
    pred=pipe.predict(Xte)
    reg_rows.append({"model":name,"MAE":mean_absolute_error(yrte,pred),
                     "MSE":mean_squared_error(yrte,pred),
                     "RMSE":np.sqrt(mean_squared_error(yrte,pred)),"R2":r2_score(yrte,pred)})
    reg_pipes[name]=pipe
reg_cmp=pd.DataFrame(reg_rows).sort_values("RMSE")

cv=GroupKFold(n_splits=min(3,gtr.nunique()))
cs=GridSearchCV(Pipeline([("preprocess",pre),("model",RandomForestClassifier(
    class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1))]),
    {"model__n_estimators":[80],"model__max_depth":[None,12],
     "model__min_samples_leaf":[1,3]},cv=cv,scoring="f1_weighted",n_jobs=-1)
cs.fit(Xtr,yctr,groups=gtr)
rs=GridSearchCV(Pipeline([("preprocess",pre),("model",RandomForestRegressor(
    random_state=RANDOM_STATE,n_jobs=-1))]),
    {"model__n_estimators":[80],"model__max_depth":[None,12],
     "model__min_samples_leaf":[1,3]},cv=cv,scoring="neg_root_mean_squared_error",n_jobs=-1)
rs.fit(Xtr,yrtr,groups=gtr)

def irrigation_priority(data, predicted_stress):
    moisture_deficit=np.clip((.27-data["soil_moisture_m3_m3"].to_numpy())/.22,0,1)
    rain_relief=np.clip(data["rainfall_mm_day"].to_numpy()/18,0,1)
    score=np.clip(100*(.72*np.clip(predicted_stress,0,1)+.20*moisture_deficit+.08*(1-rain_relief)),0,100)
    category=pd.cut(score,[-.001,25,50,75,100.001],
        labels=["Low Priority","Moderate Priority","High Priority","Critical Priority"]).astype(str)
    return np.round(score,1),category

cls_cmp.to_csv(os.path.join(OUTPUT_DIR,"classification_comparison.csv"),index=False)
reg_cmp.to_csv(os.path.join(OUTPUT_DIR,"regression_comparison.csv"),index=False)
bundle={
 "classification_pipeline":cs.best_estimator_,"regression_pipeline":rs.best_estimator_,
 "feature_columns":X.columns.tolist(),"categorical_columns":cat_cols,"numerical_columns":num_cols,
 "classification_labels":["Low","Moderate","High","Very High"],
 "classification_metrics":cls_cmp.to_dict("records"),"regression_metrics":reg_cmp.to_dict("records"),
 "classifier_best_params":cs.best_params_,"regressor_best_params":rs.best_params_,
 "priority_method":"Illustrative rule-based score; not agronomically validated",
 "dataset_notice":"Synthetic demonstration data; not field-validated","random_state":RANDOM_STATE
}
joblib.dump(bundle,os.path.join(OUTPUT_DIR,"crop_stress_models.joblib"))
print("Training complete. Outputs:",OUTPUT_DIR)
print(cls_cmp.to_string(index=False))
print(reg_cmp.to_string(index=False))
