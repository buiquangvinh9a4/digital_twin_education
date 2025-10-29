import os
import shutil
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
SIM_DIR = os.path.join(ROOT, "data", "simulations")
MODELS = os.path.join(ROOT, "models")

def safe_clear_dir(path: str, keep: list[str] | None = None):
    os.makedirs(path, exist_ok=True)
    keep = set(keep or [])
    for name in os.listdir(path):
        if name in keep:
            continue
        p = os.path.join(path, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception:
            pass

def main():
    print("🧹 Resetting processed data, simulations, and models...")
    safe_clear_dir(PROCESSED)
    safe_clear_dir(SIM_DIR)
    # Keep nothing in models; remove all .h5 to force retrain
    if os.path.exists(MODELS):
        for name in os.listdir(MODELS):
            p = os.path.join(MODELS, name)
            try:
                if os.path.isfile(p) and (name.endswith(".h5") or name.endswith(".joblib")):
                    os.remove(p)
            except Exception:
                pass

    print("✅ Cleared. Rebuilding from OULAD...")
    # Run ETL
    subprocess.run(["python", os.path.join(ROOT, "scripts", "etl_prepare.py")], check=True)
    # Train LSTM and write predictions
    subprocess.run(["python", os.path.join(ROOT, "scripts", "train_lstm.py")], check=True)
    print("🎉 Done. You can now run simulator/update_twin or evaluation as needed.")

if __name__ == "__main__":
    main()




