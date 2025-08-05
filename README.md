
---

## ⚙️ File/Script Descriptions

### `main.py`
- Orchestrates the entire pipeline: poisons data, trains the model, and prints accuracy.
- Can be run repeatedly to test different poison levels.

### `train_model.py`
- Loads (clean or poisoned) data, trains a decision tree classifier, and saves the model in `models/`.

### `evaluate_model.py`
- Evaluates a given `.pkl` model on the full clean dataset and prints accuracy + confusion matrix.

### `test_model_accuracy.py`
- Unit test script that checks if at least one of the model versions (`v1`, `v2`, `v3`) achieves more than 70% accuracy.

---

## 🧪 Poison Scripts (Bash)

### `poison_scripts/poison_5_record.sh`
- Poisons 5% of the dataset, retrains the model, and saves it.

### `poison_scripts/poison_10.sh`
- Poisons 10% of the dataset, retrains the model, and saves it.

### `poison_scripts/poison_50.sh`
- Poisons 50% of the dataset, retrains the model, and saves it.

Each script can be run individually to simulate different attack intensities.


## ✅ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
