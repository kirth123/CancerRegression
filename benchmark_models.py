import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

avg = pd.read_csv('avg-household-size.csv')
reg = pd.read_csv('cancer-regression.csv')
df = pd.merge(avg, reg, on='geography')

features = df.drop(columns=['target_deathrate','avganncount','avgdeathsperyear','incidencerate','binnedinc']).copy()
target = df['target_deathrate']
features['state'] = features['geography'].str.split(', ').str[1]
features = features.drop(columns=['geography'])

# Feature engineering
features['pub_vs_priv'] = features['pctpubliccoverage'] - features['pctprivatecoverage']
features['public_to_private_ratio'] = features['pctpubliccoverage'] / (features['pctprivatecoverage'] + 1e-3)
features['education_ratio'] = features['pctbachdeg25_over'] / (features['pcths25_over'] + 1e-3)
features['employment_ratio'] = features['pctemployed16_over'] / (features['pctunemployed16_over'] + 1e-3)
features['age_diff'] = features['medianagemale'] - features['medianagefemale']
features['poverty_income'] = features['povertypercent'] / (features['medincome'] + 1e-3) * 1000
features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

cat_cols = ['state']
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])

models = [
    ('LinearRegression', LinearRegression()),
    ('RidgeCV', RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])),
    ('LassoCV', LassoCV(cv=5, n_alphas=100, random_state=42, max_iter=20000)),
    ('ElasticNetCV', ElasticNetCV(cv=5, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], n_alphas=80, random_state=42, max_iter=20000)),
    ('RandomForest', RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=2, random_state=42)),
    ('ExtraTrees', ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42)),
    ('GradientBoosting', GradientBoostingRegressor(random_state=42)),
    ('HistGradientBoosting', HistGradientBoostingRegressor(random_state=42)),
    ('XGBoost', XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='rmse')),
    ('SVR', SVR(C=10.0, epsilon=0.1)),
]

for name, model in models:
    pipe = Pipeline([('preprocess', preprocess), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    print(f'{name}: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}')
