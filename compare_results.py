import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import re
from datetime import datetime
from utils import save_plot, create_output_dir

def plot_confusion_matrix(y_true, y_pred, title, output_dir):
    if not y_true or not y_pred:
        print(f"No data available to plot {title} Confusion Matrix")
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title} Confusion Matrix')
    save_plot(fig, f"{title.lower().replace(' ', '_')}_confusion_matrix.png", output_dir)

def plot_roc_curve(y_true, y_pred_proba, title, output_dir):
    if not y_true or not y_pred_proba:
        print(f"No data available to plot {title} ROC Curve")
        return
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title} ROC Curve')
    ax.legend(loc="lower right")
    save_plot(fig, f"{title.lower().replace(' ', '_')}_roc_curve.png", output_dir)

def smooth_data(data, window_size=50):
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().tolist()

def plot_gpu_usage(log, title, output_dir, color='blue'):
    if 'step' not in log or 'gpu_usage' not in log:
        print(f"No data available to plot {title} GPU Usage")
        return
    fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
    smoothed_gpu_usage = smooth_data(log['gpu_usage'])
    ax.plot(log['step'], smoothed_gpu_usage, label=f'{title} GPU Usage', color=color)
    ax.set_ylim([0, 100])  # Set y-axis limit from 0 to 100
    ax.set_title(f'{title} GPU Usage')
    ax.set_xlabel('Steps')
    ax.set_ylabel('GPU Usage (%)')
    ax.legend()
    save_plot(fig, f'{title.lower().replace(" ", "_")}_gpu_usage.png', output_dir)

def plot_gpu_memory_usage(log, title, output_dir, color='blue'):
    if 'step' not in log or 'gpu_memory_usage' not in log:
        print(f"No data available to plot {title} GPU Memory Usage")
        return
    fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
    smoothed_memory_usage = smooth_data(log['gpu_memory_usage'])
    ax.plot(log['step'], smoothed_memory_usage, label=f'{title} GPU Memory Usage', color=color)
    ax.set_ylim([0, 16000])  # Adjust this limit as necessary based on your data
    ax.set_title(f'{title} GPU Memory Usage')
    ax.set_xlabel('Steps')
    ax.set_ylabel('GPU Memory Usage (MB)')
    ax.legend()
    save_plot(fig, f'{title.lower().replace(" ", "_")}_gpu_memory_usage.png', output_dir)

def plot_ram_usage(log, title, output_dir, color='blue'):
    if 'step' not in log or 'ram_usage' not in log:
        print(f"No data available to plot {title} RAM Usage")
        return
    fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
    smoothed_ram_usage = smooth_data(log['ram_usage'])
    ax.plot(log['step'], smoothed_ram_usage, label=f'{title} RAM Usage', color=color)
    ax.set_ylim([0, 32000])  # Adjust this limit as necessary based on your data
    ax.set_title(f'{title} RAM Usage')
    ax.set_xlabel('Steps')
    ax.set_ylabel('RAM Usage (MB)')
    ax.legend()
    save_plot(fig, f'{title.lower().replace(" ", "_")}_ram_usage.png', output_dir)

def plot_metric_comparison(metric_name, bert_value, elmo_value, output_dir):
    metrics_df = pd.DataFrame({
        'Metric': [metric_name],
        'BERT': [bert_value],
        'ELMo': [elmo_value]
    })
    fig, ax = plt.subplots(figsize=(10, 6), dpi=400)
    bars = metrics_df.plot(x='Metric', kind='bar', ax=ax)
    ax.set_title(f'{metric_name} Comparison')
    ax.set_ylabel(metric_name)
    
    # Annotate the bars with their values
    for p in bars.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Ensure labels are displayed horizontally
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    save_plot(fig, f'{metric_name.lower().replace(" ", "_")}_comparison.png', output_dir)

def extract_log_data(log_path):
    data = {
        'step': [],
        'gpu_usage': [],
        'gpu_memory_usage': [],
        'ram_usage': [],
        'precision': None,
        'recall': None,
        'f1_score': None,
        'training_time': None
    }

    with open(log_path, 'r') as file:
        lines = file.readlines()

    step = 0
    for line in lines:
        if "Memory Usage" in line:
            memory_usage = float(line.split(": ")[1].split()[0])
            data['ram_usage'].append(memory_usage)
        if "GPU Memory Usage" in line:
            gpu_memory_usage = float(line.split(": ")[1].split()[0])
            data['gpu_memory_usage'].append(gpu_memory_usage)
        if "GPU Usage" in line:
            gpu_usage = float(line.split(": ")[1].split()[0])
            data['gpu_usage'].append(gpu_usage)
            data['step'].append(step)
            step += 1
        if "Precision" in line:
            data['precision'] = float(line.split(": ")[1].strip())
        if "Recall" in line:
            data['recall'] = float(line.split(": ")[1].strip())
        if "F1 Score" in line:
            data['f1_score'] = float(line.split(": ")[1].strip())
        if "Training Time" in line:
            data['training_time'] = float(line.split(": ")[1].strip().replace(' seconds', ''))

    # Ensure arrays have matching lengths
    min_length = min(len(data['step']), len(data['gpu_usage']), len(data['gpu_memory_usage']), len(data['ram_usage']))
    data['step'] = data['step'][:min_length]
    data['gpu_usage'] = data['gpu_usage'][:min_length]
    data['gpu_memory_usage'] = data['gpu_memory_usage'][:min_length]
    data['ram_usage'] = data['ram_usage'][:min_length]

    return data

def find_latest_log_file(base_output_dir, model_name):
    subdirs = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]
    date_pattern = re.compile(r'\d{8}_\d{6}')
    subdirs = [d for d in subdirs if date_pattern.match(d)]

    latest_log_file = None
    latest_time = None

    for subdir in subdirs:
        log_file = os.path.join(base_output_dir, subdir, f"{model_name.lower()}_log.txt")
        if os.path.exists(log_file):
            subdir_time = datetime.strptime(subdir, '%Y%m%d_%H%M%S')
            if latest_time is None or subdir_time > latest_time:
                latest_time = subdir_time
                latest_log_file = log_file

    if not latest_log_file:
        raise FileNotFoundError(f"{model_name} log file not found in any subdirectory.")

    return latest_log_file

def find_csv_file(log_file):
    base_dir = os.path.dirname(log_file)
    csv_file = os.path.join(base_dir, f"{os.path.basename(log_file).split('_')[0]}_sentiment_predictions.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found for log file: {log_file}")
    return csv_file

def plot_combined_metric(bert_log, elmo_log, title, ylabel, filename, output_dir, metric_key):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
    smoothed_bert_metric = smooth_data(bert_log[metric_key])
    smoothed_elmo_metric = smooth_data(elmo_log[metric_key])
    ax.plot(bert_log['step'], smoothed_bert_metric, label='BERT', color='blue')
    ax.plot(elmo_log['step'], smoothed_elmo_metric, label='ELMo', color='orange')
    if ylabel == 'GPU Usage (%)':
        ax.set_ylim([0, 100])  # Set y-axis limit from 0 to 100 for percentage plots
    ax.set_title(f'{title} Comparison')
    ax.set_xlabel('Steps')
    ax.set_ylabel(ylabel)
    ax.legend()
    save_plot(fig, f'{filename}.png', output_dir)

def main():
    base_output_dir = "/mnt/c/Users/cleme/OneDrive/Desktop/Datenanalyse"
    output_dir = create_output_dir(base_output_dir)

    bert_log_file = find_latest_log_file(base_output_dir, "bert")
    elmo_log_file = find_latest_log_file(base_output_dir, "elmo")

    bert_log = extract_log_data(bert_log_file)
    elmo_log = extract_log_data(elmo_log_file)

    bert_csv_file = find_csv_file(bert_log_file)
    elmo_csv_file = find_csv_file(elmo_log_file)

    bert_csv_data = pd.read_csv(bert_csv_file)
    elmo_csv_data = pd.read_csv(elmo_csv_file)

    # Update the column names based on the actual CSV files
    bert_log['y_true'] = bert_csv_data['Actual Sentiment'].tolist()
    bert_log['y_pred'] = bert_csv_data['Predicted Sentiment'].tolist()

    elmo_log['y_true'] = elmo_csv_data['Actual Sentiment'].tolist()
    elmo_log['y_pred'] = elmo_csv_data['Predicted Sentiment'].tolist()

    # Plotting the GPU, RAM usages, and metric comparisons
    plot_gpu_usage(bert_log, 'BERT', output_dir, color='blue')
    plot_gpu_usage(elmo_log, 'ELMo', output_dir, color='orange')
    plot_gpu_memory_usage(bert_log, 'BERT', output_dir, color='blue')
    plot_gpu_memory_usage(elmo_log, 'ELMo', output_dir, color='orange')
    plot_ram_usage(bert_log, 'BERT', output_dir, color='blue')
    plot_ram_usage(elmo_log, 'ELMo', output_dir, color='orange')

    plot_metric_comparison('Precision', bert_log['precision'], elmo_log['precision'], output_dir)
    plot_metric_comparison('Recall', bert_log['recall'], elmo_log['recall'], output_dir)
    plot_metric_comparison('F1 Score', bert_log['f1_score'], elmo_log['f1_score'], output_dir)
    plot_metric_comparison('Training Time', bert_log['training_time'], elmo_log['training_time'], output_dir)
    
    # Plot confusion matrices and ROC curves
    plot_confusion_matrix(bert_log['y_true'], bert_log['y_pred'], 'BERT', output_dir)
    plot_confusion_matrix(elmo_log['y_true'], elmo_log['y_pred'], 'ELMo', output_dir)

    # If 'Predicted Probability' column is present
    if 'Predicted Probability' in bert_csv_data.columns:
        bert_log['y_pred_proba'] = bert_csv_data['Predicted Probability'].tolist()
        plot_roc_curve(bert_log['y_true'], bert_log['y_pred_proba'], 'BERT', output_dir)

    if 'Predicted Probability' in elmo_csv_data.columns:
        elmo_log['y_pred_proba'] = elmo_csv_data['Predicted Probability'].tolist()
        plot_roc_curve(elmo_log['y_true'], elmo_log['y_pred_proba'], 'ELMo', output_dir)

    # Additional comparison plots
    plot_combined_metric(bert_log, elmo_log, 'RAM Usage', 'RAM Usage (MB)', 'ram_usage_comparison', output_dir, 'ram_usage')
    plot_combined_metric(bert_log, elmo_log, 'GPU Memory Usage', 'GPU Memory Usage (MB)', 'gpu_memory_usage_comparison', output_dir, 'gpu_memory_usage')
    plot_combined_metric(bert_log, elmo_log, 'GPU Usage', 'GPU Usage (%)', 'gpu_usage_comparison', output_dir, 'gpu_usage')

if __name__ == "__main__":
    main()
