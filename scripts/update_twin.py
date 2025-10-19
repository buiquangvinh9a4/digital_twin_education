import os, time, subprocess

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/oulad_lstm.h5"
PRED_PATH = "data/processed/ou_pred.csv"

def file_changed(file, last_mtime):
    try:
        return os.path.getmtime(file) != last_mtime
    except:
        return False

if __name__ == "__main__":
    last_mtime = 0
    print("👁‍🗨 Watching test.csv for changes...")
    while True:
        if file_changed(TEST_PATH, last_mtime):
            print("📡 Detected change! Re-running model...")
            subprocess.run(["python", "scripts/train_model.py"])
            last_mtime = os.path.getmtime(TEST_PATH)
        time.sleep(5)
