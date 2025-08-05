import joblib
from sklearn.linear_model import LinearRegression

def train_meta_model(X_meta, y_meta, save_path):
    meta_model = LinearRegression()
    meta_model.fit(X_meta, y_meta)
    joblib.dump(meta_model, save_path)
    return meta_model

def load_meta_model(path):
    return joblib.load(path)
