import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../CONFOLD/')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import numpy as np
from foldrm import Classifier
from utils import split_data_stratified
from datasets import rk_mammals
from algo import prune_rules

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
model_template, data = rk_mammals()

# Split into training and testing sets
train_data, test_data = split_data_stratified(data, ratio=0.80, shuffle=True)

print(f"Training set size: {len(train_data)} rk_mammals")
print(f"Testing set size: {len(test_data)} rk_mammals")

# Prepare the test data
X_test = [d[:-1] for d in test_data]
Y_test = [d[-1] for d in test_data]

# Store all predictions
all_predictions = {}

# ============================================================
# 1. BASELINE MODEL
# ============================================================
baseline_model = Classifier(attrs=model_template.attrs.copy(), numeric=model_template.numeric, label=model_template.label)
baseline_model.fit(train_data, ratio=0.5)

print("\n" + "="*70)
print("BASELINE MODEL")
print("="*70)
print("\n--- Rules Learned by the Baseline Model ---")
baseline_model.print_asp(simple=True)

# Get predictions
predictions_tuples = baseline_model.predict(X_test)
predicted_labels = [p[0] for p in predictions_tuples]
all_predictions['baseline'] = predicted_labels

# Calculate accuracy
accuracy = sum(1 for i in range(len(Y_test)) if predicted_labels[i] == Y_test[i]) / len(Y_test)
print(f"\nBaseline Accuracy: {accuracy * 100:.2f}%")

# Save baseline results
os.makedirs('confold_results_rk', exist_ok=True)
with open('confold_results_rk/01_baseline.txt', 'w') as f:
    baseline_model.asp()
    f.write("BASELINE MODEL\n" + "="*70 + "\n\n")
    f.write("RULES:\n" + "-"*70 + "\n")
    f.write("\n".join(baseline_model.asp_rules) + "\n\n")
    
    def _n(x): return 'None' if x is None else str(x).strip()
    y_true_n = [_n(y) for y in Y_test]
    y_pred_n = [_n(y) for y in predicted_labels]
    labels_c, mat_c, metrics_c = confusion_matrix_and_metrics(y_true_n, y_pred_n)
    
    f.write("="*70 + "\nPERFORMANCE METRICS\n" + "="*70 + "\n")
    header = [""] + [f"PRED:{l}" for l in labels_c]
    rows = [[f"TRUE:{labels_c[i]}"] + mat_c[i] for i in range(len(labels_c))]
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([header] + rows))]
    f.write("\nConfusion Matrix:\n")
    f.write(" ".join(str(x).rjust(w) for x, w in zip(header, col_widths)) + "\n")
    for row in rows:
        f.write(" ".join(str(x).rjust(w) for x, w in zip(row, col_widths)) + "\n")
    
    f.write("\nPer-class Metrics:\n" + "-"*70 + "\n")
    for lbl in labels_c:
        m = metrics_c[lbl]
        f.write(f"{lbl}:\n")
        f.write(f"  Precision: {m['precision']:.3f}\n")
        f.write(f"  Recall:    {m['recall']:.3f}\n")
        f.write(f"  F1-Score:  {m['f1']:.3f}\n")
        f.write(f"  Support:   {m['support']}\n\n")
    
    f.write("-"*70 + f"\nOverall Accuracy: {accuracy * 100:.2f}%\n" + "="*70 + "\n")

# ============================================================
# 2. SIMPLE PRUNING (Post-Hoc Confidence Pruning)
# ============================================================
print("\n" + "="*70)
print("SIMPLE PRUNING")
print("="*70)

# Apply pruning to baseline rules
pruned_rules = prune_rules(baseline_model.rules, confidence=0.70)

# Create new model with pruned rules
simple_pruned_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)
simple_pruned_model.rules = pruned_rules

print("\n--- Rules After Simple Pruning (Confidence >= 0.90) ---")
simple_pruned_model.print_asp(simple=True)

# Get predictions
predictions_simple_pruned = simple_pruned_model.predict(X_test)
predicted_labels_simple_pruned = [p[0] for p in predictions_simple_pruned]
all_predictions['simple_pruning'] = predicted_labels_simple_pruned

# Calculate accuracy
simple_pruned_accuracy = sum(1 for i in range(len(Y_test)) if predicted_labels_simple_pruned[i] == Y_test[i]) / len(Y_test)
print(f"\nSimple Pruning Accuracy: {simple_pruned_accuracy * 100:.2f}%")

# Save results
with open('confold_results_rk/02_simple_pruned_model.txt', 'w') as f:
    simple_pruned_model.asp()
    f.write("SIMPLE PRUNING (Confidence >= 0.90)\n" + "="*70 + "\n\n")
    f.write("RULES:\n" + "-"*70 + "\n")
    f.write("\n".join(simple_pruned_model.asp_rules) + "\n\n")
    
    def _n(x): return 'None' if x is None else str(x).strip()
    y_true_n = [_n(y) for y in Y_test]
    y_pred_n = [_n(y) for y in predicted_labels_simple_pruned]
    labels_c, mat_c, metrics_c = confusion_matrix_and_metrics(y_true_n, y_pred_n)
    
    f.write("="*70 + "\nPERFORMANCE METRICS\n" + "="*70 + "\n")
    header = [""] + [f"PRED:{l}" for l in labels_c]
    rows = [[f"TRUE:{labels_c[i]}"] + mat_c[i] for i in range(len(labels_c))]
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([header] + rows))]
    f.write("\nConfusion Matrix:\n")
    f.write(" ".join(str(x).rjust(w) for x, w in zip(header, col_widths)) + "\n")
    for row in rows:
        f.write(" ".join(str(x).rjust(w) for x, w in zip(row, col_widths)) + "\n")
    
    f.write("\nPer-class Metrics:\n" + "-"*70 + "\n")
    for lbl in labels_c:
        m = metrics_c[lbl]
        f.write(f"{lbl}:\n")
        f.write(f"  Precision: {m['precision']:.3f}\n")
        f.write(f"  Recall:    {m['recall']:.3f}\n")
        f.write(f"  F1-Score:  {m['f1']:.3f}\n")
        f.write(f"  Support:   {m['support']}\n\n")
    
    f.write("-"*70 + f"\nOverall Accuracy: {simple_pruned_accuracy * 100:.2f}%\n" + "="*70 + "\n")

# ============================================================
# 3. ADVANCED PRUNING (Confidence-Driven Learning)
# ============================================================
print("\n" + "="*70)
print("ADVANCED PRUNING")
print("="*70)

# Create new model for advanced pruning
advanced_pruning_model = Classifier(attrs=model_template.attrs.copy(), numeric=model_template.numeric, label=model_template.label)

# Train using confidence_fit with high improvement threshold
print("\n--- Training with confidence_fit(improvement_threshold=0.10) ---")
advanced_pruning_model.confidence_fit(train_data, improvement_threshold=0.1)

print("\n--- Rules Learned via Confidence-Driven Learning ---")
advanced_pruning_model.print_asp(simple=True)

# Get predictions
predictions_advanced = advanced_pruning_model.predict(X_test)
predicted_labels_advanced = [p[0] for p in predictions_advanced]
all_predictions['advanced_pruning'] = predicted_labels_advanced

# Calculate accuracy
advanced_accuracy = sum(1 for i in range(len(Y_test)) if predicted_labels_advanced[i] == Y_test[i]) / len(Y_test)
print(f"\nAdvanced Pruning Accuracy: {advanced_accuracy * 100:.2f}%")

# Save results
with open('confold_results_rk/03_advanced_pruning_model.txt', 'w') as f:
    advanced_pruning_model.asp()
    f.write("ADVANCED PRUNING\n" + "="*70 + "\n\n")
    f.write("RULES:\n" + "-"*70 + "\n")
    f.write("\n".join(advanced_pruning_model.asp_rules) + "\n\n")
    
    def _n(x): return 'None' if x is None else str(x).strip()
    y_true_n = [_n(y) for y in Y_test]
    y_pred_n = [_n(y) for y in predicted_labels_advanced]
    labels_c, mat_c, metrics_c = confusion_matrix_and_metrics(y_true_n, y_pred_n)
    
    f.write("="*70 + "\nPERFORMANCE METRICS\n" + "="*70 + "\n")
    header = [""] + [f"PRED:{l}" for l in labels_c]
    rows = [[f"TRUE:{labels_c[i]}"] + mat_c[i] for i in range(len(labels_c))]
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([header] + rows))]
    f.write("\nConfusion Matrix:\n")
    f.write(" ".join(str(x).rjust(w) for x, w in zip(header, col_widths)) + "\n")
    for row in rows:
        f.write(" ".join(str(x).rjust(w) for x, w in zip(row, col_widths)) + "\n")
    
    f.write("\nPer-class Metrics:\n" + "-"*70 + "\n")
    for lbl in labels_c:
        m = metrics_c[lbl]
        f.write(f"{lbl}:\n")
        f.write(f"  Precision: {m['precision']:.3f}\n")
        f.write(f"  Recall:    {m['recall']:.3f}\n")
        f.write(f"  F1-Score:  {m['f1']:.3f}\n")
        f.write(f"  Support:   {m['support']}\n\n")
    
    f.write("-"*70 + f"\nOverall Accuracy: {advanced_accuracy * 100:.2f}%\n" + "="*70 + "\n")

# ============================================================
# CONSOLIDATED RESULTS
# ============================================================
def _norm_label(x):
    if x is None:
        return 'None'
    return str(x).strip()

print('\n' + "="*70)
print('CONSOLIDATED CONFUSION MATRICES AND METRICS')
print("="*70)

Y_test_norm = [_norm_label(y) for y in Y_test]

for key, y_pred in [('Baseline', all_predictions.get('baseline')),
                     ('Simple Pruning', all_predictions.get('simple_pruning')),
                     ('Advanced Pruning', all_predictions.get('advanced_pruning'))]:
    if y_pred is None:
        continue
    y_pred_norm = [_norm_label(y) for y in y_pred]
    labels_c, mat_c, metrics_c = confusion_matrix_and_metrics(Y_test_norm, y_pred_norm)
    print(f"\n--- {key} ---")
    print_confusion_matrix(labels_c, mat_c)
    print('\nPer-class metrics:')
    for lbl in labels_c:
        m = metrics_c[lbl]
        print(f"{lbl}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}, support={m['support']}")
    total_correct = sum(metrics_c[l]['TP'] for l in labels_c)
    total = sum(metrics_c[l]['support'] for l in labels_c)
    acc = total_correct / total if total > 0 else 0.0
    print(f"Overall accuracy: {acc * 100:.2f}%")