from data_utils import load_data
from stacking_pipeline import train_full_pipeline, fine_tune_full_pipeline

def main():
    train_path = "train_dataset.csv"
    finetune_path = "new_dataset.csv"
    base_save_dir = "base_models"
    nn_pretrain_path = "nn_pretrained.h5"
    nn_finetune_path = "nn_finetuned.h5"
    meta_pretrain_path = "meta_pretrained.joblib"
    meta_finetune_path = "meta_finetuned.joblib"

    X_train, y_train = load_data(train_path)
    X_finetune, y_finetune = load_data(finetune_path)

    train_full_pipeline(X_train, y_train,
                        base_save_dir, nn_pretrain_path,
                        meta_pretrain_path)
    print("Pretraining complete.")

    base_models, nn_finetune, meta_model, rmse = fine_tune_full_pipeline(
        X_finetune, y_finetune,
        base_save_dir, nn_pretrain_path,
        nn_finetune_path, meta_pretrain_path,
        meta_finetune_path)

    print(f"Transfer learning RMSE on fine-tune set: {rmse:.6f}")

if __name__ == "__main__":
    main()
