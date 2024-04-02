from tabulate import tabulate
import pandas as pd

# Metrics data
models_metrics = {
    'Main Model': {
        'Accuracy': 0.6000,
        'Precision': [0.8214, 0.6000, 0.4194, 0.0000],
        'Mean Recall': 0.4873,
        'F1_measure': 0.4734
    },
    'Variant 1 Model': {
        'Accuracy': 0.5692,
        'Precision': [0.8462, 0.2667, 0.5000, 0.4286],
        'Mean Recall': 0.5101,
        'F1_measure': 0.5102
    },
    'Variant 2 Model': {
        'Accuracy': 0.6923,
        'Precision': [0.8966, 0.5714, 0.4762, 0.6250],
        'Mean Recall': 0.6012,
        'F1_measure': 0.6211
    }
}

# Compute the macro-averaged and micro-averaged metrics for each model
macro_averaged_precision = []
macro_averaged_recall = []
macro_averaged_f1_score = []
micro_averaged_precision = []
micro_averaged_recall = []
micro_averaged_f1_score = []

for model_name, model_metrics in models_metrics.items():
    # Compute the macro-averaged metrics
    macro_precision = sum(model_metrics['Precision']) / len(model_metrics['Precision'])
    macro_recall = model_metrics['Mean Recall']
    macro_f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    # Compute the micro-averaged metrics
    true_positives = sum(model_metrics['Precision'])
    false_positives = sum(model_metrics['Precision']) - sum(precision ** 2 for precision in model_metrics['Precision'])
    false_negatives = len(model_metrics['Precision']) - true_positives - false_positives
    micro_precision = true_positives / (true_positives + false_positives)
    micro_recall = true_positives / (true_positives + false_negatives)
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    # Append the computed metrics to the corresponding lists
    macro_averaged_precision.append(macro_precision)
    macro_averaged_recall.append(macro_recall)
    macro_averaged_f1_score.append(macro_f1_score)
    micro_averaged_precision.append(micro_precision)
    micro_averaged_recall.append(micro_recall)
    micro_averaged_f1_score.append(micro_f1_score)

# Define the results for each model
results = {
    "Model": [],
    "Accuracy": [],
    "Macro-averaged Precision": [],
    "Macro-averaged Recall": [],
    "Macro-averaged F1-score": [],
    "Micro-averaged Precision": [],
    "Micro-averaged Recall": [],
    "Micro-averaged F1-score": [],
}

for i, (model_name, model_metrics) in enumerate(models_metrics.items()):
    results["Model"].append(model_name)
    results["Accuracy"].append(model_metrics['Accuracy'])
    results["Macro-averaged Precision"].append(macro_averaged_precision[i])
    results["Macro-averaged Recall"].append(macro_averaged_recall[i])
    results["Macro-averaged F1-score"].append(macro_averaged_f1_score[i])
    results["Micro-averaged Precision"].append(micro_averaged_precision[i])
    results["Micro-averaged Recall"].append(micro_averaged_recall[i])
    results["Micro-averaged F1-score"].append(micro_averaged_f1_score[i])

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Display the table
print("\n\n\nTABLE 1\n")
print(df.to_markdown())



# Calculate micro-averaged metrics
total_samples = 65  # Assuming a total of 100 samples for demonstration purposes
for model, metrics in models_metrics.items():
    metrics['Macro-Averaged Precision'] = sum(metrics['Precision']) / len(metrics['Precision'])
    metrics['Micro-Averaged Precision'] = sum(metrics['Precision']) / len(metrics['Precision'])
    metrics['Micro-Averaged Recall'] = metrics['Mean Recall']
    metrics['Micro-Averaged F1'] = 2 * (metrics['Micro-Averaged Precision'] * metrics['Micro-Averaged Recall']) / \
                                    (metrics['Micro-Averaged Precision'] + metrics['Micro-Averaged Recall'])

# Prepare data for the table
table_data = []
for model, metrics in models_metrics.items():
    table_row = [model,
                 metrics['Macro-Averaged Precision'], metrics['Micro-Averaged Recall'], metrics['Micro-Averaged F1'],
                 metrics['Precision'], metrics['Mean Recall'], metrics['F1_measure'],
                 metrics['Accuracy']]
    table_data.append(table_row)

# Table headers
headers = ['Model', 'Macro-Precision', 'Micro-Recall', 'Micro-F1', 'Precision', 'Macro-Recall', 'Macro-F1',
           'Accuracy']

# Generate table
table = tabulate(table_data, headers=headers, tablefmt="pretty")

# Print table
print("\n\n\nTABLE 2\n")
print(table)
