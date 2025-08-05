import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import os

def train_base_models(X_train, y_train):
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(X_train, y_train)
    svr = SVR(kernel='rbf', C=1000, epsilon=0.01)
    svr.fit(X_train, y_train)
    gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=30, random_state=42)
    gbr.fit(X_train, y_train)
    knnr = KNeighborsRegressor(n_neighbors=105, leaf_size=30, p=12)
    knnr.fit(X_train, y_train)
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.001, max_depth=100,
                      use_label_encoder=False, eval_metric='rmse')
    xgb.fit(X_train, y_train)
    return {'rfr': rfr, 'svr': svr, 'gbr': gbr, 'knnr': knnr, 'xgb': xgb}

def save_base_models(base_models, base_save_dir):
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    for name, model in base_models.items():
        joblib.dump(model, f"{base_save_dir}/{name}.joblib")

def load_base_models(base_save_dir):
    base_models = {}
    for name in ['rfr', 'svr', 'gbr', 'knnr', 'xgb']:
        base_models[name] = joblib.load(f"{base_save_dir}/{name}.joblib")
    return base_models
