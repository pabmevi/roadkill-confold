import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../CONFOLD/')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import numpy as np
from foldrm import Classifier
from utils import split_data
from datasets import final_extinctionrisk

# ---------------- seed / reproducibility ----------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
args, _ = parser.parse_known_args()
SEED = args.seed if args.seed is not None else int(os.environ.get('SEED', '42'))
random.seed(SEED)
np.random.seed(SEED)
print(f"Using seed={SEED}")

# --- Confusion matrix helper ---
def confusion_matrix_and_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    index = {label: i for i, label in enumerate(labels)}
    C = len(labels)
    mat = [[0] * C for _ in range(C)]
    for yt, yp in zip(y_true, y_pred):
        if yp is None:
            yp = 'None'
        if yt not in index or yp not in index:
            continue
        mat[index[yt]][index[yp]] += 1
    metrics = {}
    for i, label in enumerate(labels):
        TP = mat[i][i]
        FN = sum(mat[i][j] for j in range(C) if j != i)
        FP = sum(mat[r][i] for r in range(C) if r != i)
        TN = sum(mat[r][c] for r in range(C) for c in range(C)) - TP - FP - FN
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = TP + FN
        metrics[label] = {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
                          "precision": precision, "recall": recall, "f1": f1, "support": support}
    return labels, mat, metrics

def print_confusion_matrix(labels, mat):
    header = [""] + [f"PRED:{l}" for l in labels]
    rows = []
    for i, l in enumerate(labels):
        rows.append([f"TRUE:{l}"] + mat[i])
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([header] + rows))]
    print("\nConfusion matrix:")
    print(" ".join(str(x).rjust(w) for x, w in zip(header, col_widths)))
    for row in rows:
        print(" ".join(str(x).rjust(w) for x, w in zip(row, col_widths)))

# Load the data
model_template, data = final_extinctionrisk()

# Verify the label is correct
print(f"Label column: {model_template.label}")
if model_template.label != 'extinction_risk':
    print(f"WARNING: Label is '{model_template.label}', expected 'extinction_risk'")
    print("Please check your datasets.py file")

# ============================================================================
# 30 REPETICIONES CON 80/20 SPLIT
# ============================================================================

n_repetitions = 30
results = {
    'baseline': [],
    'expert_with_confidence': [],
    'expert_no_confidence': [],
    'advanced_pruning': []
}

print(f"\n{'='*60}")
print(f"RUNNING {n_repetitions} REPETITIONS WITH 80/20 SPLIT")
print(f"{'='*60}\n")

for rep in range(n_repetitions):
    print(f"\n--- Repetition {rep + 1}/{n_repetitions} ---")
    
    # Split data (80% train, 20% test)
    train_data, test_data = split_data(data, ratio=0.80, shuffle=True)
    
    print(f"Training: {len(train_data)} | Testing: {len(test_data)}")
    
    # Prepare test data
    X_test = [d[:-1] for d in test_data]
    Y_test = [d[-1] for d in test_data]
    
    # ==================== BASELINE MODEL ====================
    baseline_model = Classifier(attrs=model_template.attrs.copy(), 
                                numeric=model_template.numeric, 
                                label=model_template.label)
    baseline_model.fit(train_data, ratio=0.5)
    
    predictions_baseline = baseline_model.predict(X_test)
    predicted_labels_baseline = [p[0] for p in predictions_baseline]
    accuracy_baseline = sum(1 for i in range(len(Y_test)) 
                           if predicted_labels_baseline[i] == Y_test[i]) / len(Y_test)
    results['baseline'].append(accuracy_baseline)
    
    # ==================== EXPERT MODEL (WITH CONFIDENCE) ====================
    expert_model = Classifier(attrs=model_template.attrs.copy(), 
                             numeric=model_template.numeric, 
                             label=model_template.label)
    
    # Expert rules with confidence
    rule1 = "with confidence 0.99 class = 'Higher_risk' if 'Range_size' '<=' '50000'"
    rule2 = "with confidence 0.70 class = 'Higher_risk' if 'Body_mass' '>=' '124'"
    
    expert_model.add_manual_rule(rule1, model_template.attrs, model_template.numeric, 
                                ['Lower_risk', 'Higher_risk'], instructions=False)
    expert_model.add_manual_rule(rule2, model_template.attrs, model_template.numeric, 
                                ['Lower_risk', 'Higher_risk'], instructions=False)
    
    expert_model.fit(train_data, ratio=0.75)
    
    predictions_expert = expert_model.predict(X_test)
    predicted_labels_expert = [p[0] for p in predictions_expert]
    accuracy_expert = sum(1 for i in range(len(Y_test)) 
                         if predicted_labels_expert[i] == Y_test[i]) / len(Y_test)
    results['expert_with_confidence'].append(accuracy_expert)
    
    # ==================== EXPERT MODEL (NO CONFIDENCE) ====================
    learned_conf_model = Classifier(attrs=model_template.attrs.copy(), 
                                    numeric=model_template.numeric, 
                                    label=model_template.label)
    
    rule1_no_conf = "class = 'Higher_risk' if 'Range_size' '<=' '50000'"
    rule2_no_conf = "class = 'Higher_risk' if 'Body_mass' '>=' '124'"
    
    learned_conf_model.add_manual_rule(rule1_no_conf, model_template.attrs, 
                                       model_template.numeric, 
                                       ['Lower_risk', 'Higher_risk'], instructions=False)
    learned_conf_model.add_manual_rule(rule2_no_conf, model_template.attrs, 
                                       model_template.numeric, 
                                       ['Lower_risk', 'Higher_risk'], instructions=False)
    
    learned_conf_model.fit(train_data, ratio=0.5)
    
    predictions_learned = learned_conf_model.predict(X_test)
    predicted_labels_learned = [p[0] for p in predictions_learned]
    accuracy_learned = sum(1 for i in range(len(Y_test)) 
                          if predicted_labels_learned[i] == Y_test[i]) / len(Y_test)
    results['expert_no_confidence'].append(accuracy_learned)
    
    # ==================== ADVANCED PRUNING ====================
    advanced_model = Classifier(attrs=model_template.attrs.copy(), 
                               numeric=model_template.numeric, 
                               label=model_template.label)
    
    advanced_model.confidence_fit(train_data, improvement_threshold=0.15)
    
    predictions_advanced = advanced_model.predict(X_test)
    predicted_labels_advanced = [p[0] for p in predictions_advanced]
    accuracy_advanced = sum(1 for i in range(len(Y_test)) 
                           if predicted_labels_advanced[i] == Y_test[i]) / len(Y_test)
    results['advanced_pruning'].append(accuracy_advanced)
    
    print(f"  Baseline: {accuracy_baseline:.3f}")
    print(f"  Expert (with conf): {accuracy_expert:.3f}")
    print(f"  Expert (no conf): {accuracy_learned:.3f}")
    print(f"  Advanced pruning: {accuracy_advanced:.3f}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print(f"\n{'='*60}")
print("SUMMARY STATISTICS (30 REPETITIONS)")
print(f"{'='*60}\n")

import pandas as pd

summary_data = []
for model_name, accuracies in results.items():
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print(f"{model_name}:")
    print(f"  Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  Min: {min_acc:.3f} | Max: {max_acc:.3f}\n")
    
    summary_data.append({
        'Model': model_name,
        'Mean_Accuracy': mean_acc,
        'Std_Accuracy': std_acc,
        'Min_Accuracy': min_acc,
        'Max_Accuracy': max_acc
    })

# Save summary to CSV
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('confold_summary_30reps.csv', index=False)
print("✓ Summary saved to: confold_summary_30reps.csv")

# Save all repetitions to CSV
all_reps_data = []
for rep in range(n_repetitions):
    row = {'Repetition': rep + 1}
    for model_name in results.keys():
        row[model_name] = results[model_name][rep]
    all_reps_data.append(row)

reps_df = pd.DataFrame(all_reps_data)
reps_df.to_csv('confold_all_repetitions.csv', index=False)
print("✓ All repetitions saved to: confold_all_repetitions.csv\n")

# ============================================================================
# FINAL MODEL WITH ALL DATA (for rule inspection)
# ============================================================================

print(f"\n{'='*60}")
print("TRAINING FINAL MODELS WITH ALL DATA (for rule inspection)")
print(f"{'='*60}\n")

# Baseline final
baseline_final = Classifier(attrs=model_template.attrs.copy(), 
                           numeric=model_template.numeric, 
                           label=model_template.label)
baseline_final.fit(data, ratio=0.5)

print("--- Baseline Rules (trained on all data) ---")
baseline_final.print_asp(simple=True)

# Expert final
expert_final = Classifier(attrs=model_template.attrs.copy(), 
                         numeric=model_template.numeric, 
                         label=model_template.label)

rule1 = "with confidence 0.99 class = 'Higher_risk' if 'Range_size' '<=' '50000'"
rule2 = "with confidence 0.70 class = 'Higher_risk' if 'Body_mass' '>=' '124'"

expert_final.add_manual_rule(rule1, model_template.attrs, model_template.numeric, 
                            ['Lower_risk', 'Higher_risk'], instructions=False)
expert_final.add_manual_rule(rule2, model_template.attrs, model_template.numeric, 
                            ['Lower_risk', 'Higher_risk'], instructions=False)

expert_final.fit(data, ratio=0.75)

print("\n--- Expert Model Rules (trained on all data) ---")
expert_final.print_asp(simple=True)

print("\n✓ Analysis completed!")
print(f"\nFiles generated:")
print(f"  - confold_summary_30reps.csv (mean ± SD for each model)")
print(f"  - confold_all_repetitions.csv (all 30 accuracy values)")
print(f"\nResults also available in 'results' dictionary in memory.")