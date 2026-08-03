import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from xgboost import XGBRegressor

avg = pd.read_csv('avg-household-size.csv')
reg = pd.read_csv('cancer-regression.csv')
df = pd.merge(avg, reg, on='geography')

features = df.drop(columns=['target_deathrate','avganncount','avgdeathsperyear','incidencerate','binnedinc']).copy()
target = df['target_deathrate']
features['state'] = features['geography'].str.split(', ').str[1]
features = features.drop(columns=['geography'])

features_model = features.copy()
features_model['pub_vs_priv'] = features_model['pctpubliccoverage'] - features_model['pctprivatecoverage']
features_model['public_to_private_ratio'] = features_model['pctpubliccoverage'] / (features_model['pctprivatecoverage'] + 1e-3)
features_model['education_ratio'] = features_model['pctbachdeg25_over'] / (features_model['pcths25_over'] + 1e-3)
features_model['employment_ratio'] = features_model['pctemployed16_over'] / (features_model['pctunemployed16_over'] + 1e-3)
features_model['age_diff'] = features_model['medianagemale'] - features_model['medianagefemale']
features_model['poverty_income'] = features_model['povertypercent'] / (features_model['medincome'] + 1e-3) * 1000
features_model = features_model.replace([np.inf, -np.inf], np.nan).fillna(0)

X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(features_model, target, test_size=0.2, random_state=42)

categorical_features = ['state']
numerical_features = [col for col in features_model.columns if col not in categorical_features]

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
])

models = [
    ('RidgeCV', RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])),
    ('LassoCV', LassoCV(cv=5, random_state=42, max_iter=20000)),
    ('ElasticNetCV', ElasticNetCV(cv=5, random_state=42, max_iter=20000)),
    ('GradientBoosting', GradientBoostingRegressor(random_state=42)),
    ('XGBoost', XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='rmse'))
]

print(f"{'Model':20s} {'R2':>6s} {'RMSE':>8s} {'MAE':>8s}")
for name, model in models:
    pipe = Pipeline([('preprocess', preprocessor), ('model', model)])
    pipe.fit(X_train_model, y_train_model)
    y_pred = pipe.predict(X_test_model)
    mse = mean_squared_error(y_test_model, y_pred)
    mae = mean_absolute_error(y_test_model, y_pred)
    rmse = sqrt(mse)
    print(f"{name:20s} {r2_score(y_test_model, y_pred):6.4f} {rmse:8.2f} {mae:8.2f}")
