import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../CONFOLD/'))) #add CONFOLD to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #add the parent directory to the path

import argparse
import random
import numpy as np
from foldrm import Classifier
from utils import split_data_stratified  # Or your stratified version if you prefer
from datasets import final_extinctionrisk # Our new function

# ---------------- seed / reproducibility ----------------
# Accept a --seed CLI argument (or read SEED from env) so runs can be reproducible.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
args, _ = parser.parse_known_args()
SEED = args.seed if args.seed is not None else int(os.environ.get('SEED', '42'))
random.seed(SEED)
np.random.seed(SEED)
print(f"Using seed={SEED}")

# --- Confusion matrix helper (used for baseline and expert models) ---

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
            # skip mismatched labels
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
#data = [row for row in data if str(row[label_index]) in ['Lower_risk', 'Higher_risk']]

# Split into training and testing sets
train_data, test_data = split_data_stratified(data, ratio=0.70, shuffle=True)

print(f"Training set size: {len(train_data)} final_extinctionrisk")
print(f"Testing set size: {len(test_data)} final_extinctionrisk")

# Instantiate a new classifier for our baseline experiment
baseline_model = Classifier(attrs=model_template.attrs.copy(), numeric=model_template.numeric, label=model_template.label)

# Fit the model on the training data
baseline_model.fit(train_data, ratio=0.5)

# Print the rules the model learned
print("--- Rules Learned by the Baseline Model ---")
baseline_model.print_asp(simple=True)

# Prepare the test data (features and true labels)
X_test = [d[:-1] for d in test_data]
Y_test = [d[-1] for d in test_data]

# Get predictions (these will be tuples of (label, confidence))
predictions_tuples = baseline_model.predict(X_test)
predicted_labels = [p[0] for p in predictions_tuples]

# Calculate accuracy
accuracy = sum(1 for i in range(len(Y_test)) if predicted_labels[i] == Y_test[i]) / len(Y_test)

# Instantiate a new classifier for our expert-guided model
expert_model = Classifier(attrs=model_template.attrs.copy(), numeric=model_template.numeric, label=model_template.label)

# Define our expert rules as strings
# Note: the symbols '==' and '<=' must also be in single quotes for the parser.
rule1 = "with confidence 0.90 class = 'Higher_risk' if 'Range_size' '<=' '130822'" #This is the value of the 1st quartil of the data
#Note additional rules could be added like this:
rule2 = "with confidence 0.70 class = 'Higher_risk' if 'Body_mass' '>=' '124'"
rule3 = "with confidence 0.80 class = 'Higher_risk' if 'Biological_use_hunting' '=' '1'"
rule4 = "with confidence 0.80 class = 'Higher_risk' if 'Agriculture' '=' '1'"

# Add the manual rules to the model
expert_model.add_manual_rule(rule1, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)
# Note: here is code to add an additional rule:
expert_model.add_manual_rule(rule2, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)
expert_model.add_manual_rule(rule3, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)
expert_model.add_manual_rule(rule4, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)

print("--- Manual Rules Added to the Model (Before Training) ---")
# The internal representation is a bit complex, but we can see our rules are in there.
for rule in expert_model.rules:
    print(rule)

# Now, fit the model on the training data.
# The algorithm will work around the rules we provided.
expert_model.fit(train_data, ratio=0.75)

# Print the final, combined rule set
print("--- Final Ruleset from the Expert Model ---")
expert_model.print_asp(simple=True)

# Get predictions from our new model
expert_predictions_tuples = expert_model.predict(X_test)
expert_predicted_labels = [p[0] for p in expert_predictions_tuples]

# Calculate accuracy
expert_accuracy = sum(1 for i in range(len(Y_test)) if expert_predicted_labels[i] == Y_test[i]) / len(Y_test)

print("--- Baseline Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("--- Expert Model Evaluation ---")
print(f"Accuracy: {expert_accuracy * 100:.2f}%")

# Store predictions so we can print confusion matrices at the end of the script
all_predictions = {}
all_predictions['baseline'] = predicted_labels
all_predictions['expert_with_confidence'] = expert_predicted_labels

# Instantiate a new classifier
learned_confidence_model = Classifier(attrs=model_template.attrs.copy(), numeric=model_template.numeric, label=model_template.label)

# Define our expert rules as strings, but WITHOUT the 'with confidence' part.
rule1_no_confidence = "class = 'Higher_risk' if 'Range_size' '<=' '130822'"
rule2_no_confidence = "class = 'Higher_risk' if 'Body_mass' '>=' '124'"
rule3_no_confidence = "class = 'Higher_risk' if 'Biological_use_hunting' '=' '1'"
rule4_no_confidence = "class = 'Higher_risk' if 'Agriculture' '=' '1'"

# Add the manual rules to the model
learned_confidence_model.add_manual_rule(rule1_no_confidence, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)
learned_confidence_model.add_manual_rule(rule2_no_confidence, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)
learned_confidence_model.add_manual_rule(rule3_no_confidence, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)
learned_confidence_model.add_manual_rule(rule4_no_confidence, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)

print("--- Manual Rules Added (Before Training) ---")
print("Notice the default confidence value of 0.5 assigned to each rule.")
for rule in learned_confidence_model.rules:
    print(rule)

# Now, fit the model on the training data.
# The algorithm will calculate the confidence of our provided rules and then learn any additional rules needed.
learned_confidence_model.fit(train_data, ratio=0.5)

# Print the final, combined rule set
print("--- Final Ruleset with Learned Confidence ---")
print("The confidence values have now been updated based on the training data!")
learned_confidence_model.print_asp(simple=True)
            #Note that confidence values will be relatively low due to the small size of the training data. 

# Get predictions from our new model
learned_conf_predictions = learned_confidence_model.predict(X_test)
learned_conf_labels = [p[0] for p in learned_conf_predictions]

# Calculate accuracy
learned_conf_accuracy = sum(1 for i in range(len(Y_test)) if learned_conf_labels[i] == Y_test[i]) / len(Y_test)

print(f"Accuracy: {learned_conf_accuracy * 100:.2f}%")

# Keep learned-confidence predictions too (this is the expert rules without an explicit confidence)
all_predictions['expert_no_confidence'] = learned_conf_labels

# First, let's re-print the rules from our expert model for comparison
print("--- Rules Before Pruning ---")
expert_model.print_asp(simple=True)

############PRUNNING##################

# Method 1: Simple Post-Hoc Confidence Pruning: removing those rules with a low confidence according to me
# Import the prune_rules function from the core algorithm file
#from algo import prune_rules

# Apply the pruning function
# This will create a new list containing only the rules that meet the confidence threshold.
#pruned_rules = prune_rules(expert_model.rules, confidence=0.90)

# We can create a new model instance to hold these pruned rules
#simple_pruned_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)
#simple_pruned_model.rules = pruned_rules

#print("\n--- Rules After Pruning (Confidence >= 0.90) ---")
#simple_pruned_model.print_asp(simple=True)
            
# Instantiate a new model for this experiment
advanced_pruning_model = Classifier(attrs=model_template.attrs.copy(), numeric=model_template.numeric, label=model_template.label)

##################
#### Method 2: Advanced Confidence-Driven Learning

# Now, train using confidence_fit with a high 15% improvement threshold
print("--- Training with confidence_fit(improvement_threshold=0.15) ---")
advanced_pruning_model.confidence_fit(train_data, improvement_threshold=0.15)

print("\n--- Rules Learned via Confidence-Driven Learning ---")
print("Note how the model is simpler and did not learn any exceptions to rules or `abnormalities', as they did not meet the high confidence improvement threshold.")
advanced_pruning_model.print_asp(simple=True)

# === Predictions for Advanced Pruning Model ===
predictions_advanced = advanced_pruning_model.predict(X_test)
predicted_labels_advanced = [p[0] for p in predictions_advanced]

# Store predictions so the consolidated metrics can include this model
all_predictions['advanced_pruning'] = predicted_labels_advanced


# ------------------ Consolidated Confusion Matrices (end of script) ------------------
def _norm_label(x):
    # Normalize labels to comparable strings
    if x is None:
        return 'None'
    return str(x).strip()

print('\n=== Consolidated Confusion Matrices and Metrics ===')
Y_test_norm = [_norm_label(y) for y in Y_test]

for key, y_pred in [('Baseline', all_predictions.get('baseline')),
                     ('Expert (rule confidence provided)', all_predictions.get('expert_with_confidence')),
                     ('Expert (without providing rule confidence)', all_predictions.get('expert_no_confidence')),
                     ('Advanced pruning', all_predictions.get('advanced_pruning'))]:
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
    # Overall accuracy
    total_correct = sum(metrics_c[l]['TP'] for l in labels_c)
    total = sum(metrics_c[l]['support'] for l in labels_c)
    acc = total_correct / total if total > 0 else 0.0
    print(f"Overall accuracy: {acc * 100:.2f}%")
