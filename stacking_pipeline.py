from base_models import train_base_models, save_base_models, load_base_models
from nn_model import train_nn, fine_tune_nn, load_nn_model
from meta_model import train_meta_model, load_meta_model
import numpy as np
from sklearn.metrics import mean_squared_error

def generate_meta_features(base_models, nn_model, X):
    preds = []
    for name, model in base_models.items():
        preds.append(model.predict(X).reshape(-1,1))
    preds.append(nn_model.predict(X).reshape(-1,1))
    return np.hstack(preds)

def train_full_pipeline(X_train, y_train,
                        base_save_dir, nn_pretrain_path,
                        meta_pretrain_path):
    base_models = train_base_models(X_train, y_train)
    save_base_models(base_models, base_save_dir)
    nn_model = train_nn(X_train, y_train, lr=0.001, epochs=1000, save_path=nn_pretrain_path)
    X_meta_train = generate_meta_features(base_models, nn_model, X_train)
    meta_model = train_meta_model(X_meta_train, y_train, meta_pretrain_path)
    return base_models, nn_model, meta_model

def fine_tune_full_pipeline(X_finetune, y_finetune,
                            base_save_dir, nn_pretrain_path,
                            nn_finetune_path, meta_pretrain_path,
                            meta_finetune_path):
    base_models = load_base_models(base_save_dir)
    nn_model_finetune = fine_tune_nn(nn_pretrain_path, X_finetune, y_finetune,
                                     lr=0.0001, epochs=1000, save_path=nn_finetune_path)
    meta_model = load_meta_model(meta_pretrain_path)
    X_meta_finetune = generate_meta_features(base_models, nn_model_finetune, X_finetune)
    meta_model.fit(X_meta_finetune, y_finetune)
    import joblib
    joblib.dump(meta_model, meta_finetune_path)
    preds = meta_model.predict(X_meta_finetune)
    rmse = np.sqrt(mean_squared_error(y_finetune, preds))
    return base_models, nn_model_finetune, meta_model, rmse
