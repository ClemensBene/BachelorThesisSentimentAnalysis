import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(fig, filename):
    output_dir = "/mnt/c/Users/cleme/OneDrive/Desktop/Bachelorarbeit Sentiment/Code/comparison_pics"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    print(f"Saved plot to {filepath}")

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title} Confusion Matrix')
    save_plot(fig, f"{title.lower().replace(' ', '_')}_confusion_matrix.png")

def plot_roc_curve(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title} ROC Curve')
    ax.legend(loc="lower right")
    save_plot(fig, f"{title.lower().replace(' ', '_')}_roc_curve.png")

def compare_predictions(elmo_df, bert_df):
    if not (elmo_df['Review'].equals(bert_df['Review'])):
        raise ValueError("The reviews in the ELMo and BERT datasets do not match.")
    
    comparison_df = pd.DataFrame({
        'Review': elmo_df['Review'],
        'Actual Sentiment': elmo_df['Actual Sentiment'],
        'ELMo Predicted Sentiment': elmo_df['Predicted Sentiment'],
        'BERT Predicted Sentiment': bert_df['Predicted Sentiment']
    })

    # Calculate metrics
    accuracy_elmo = (comparison_df['Actual Sentiment'] == comparison_df['ELMo Predicted Sentiment']).mean() * 100
    accuracy_bert = (comparison_df['Actual Sentiment'] == comparison_df['BERT Predicted Sentiment']).mean() * 100

    precision_elmo = precision_score(comparison_df['Actual Sentiment'], comparison_df['ELMo Predicted Sentiment'])
    precision_bert = precision_score(comparison_df['Actual Sentiment'], comparison_df['BERT Predicted Sentiment'])

    recall_elmo = recall_score(comparison_df['Actual Sentiment'], comparison_df['ELMo Predicted Sentiment'])
    recall_bert = recall_score(comparison_df['Actual Sentiment'], comparison_df['BERT Predicted Sentiment'])

    f1_elmo = f1_score(comparison_df['Actual Sentiment'], comparison_df['ELMo Predicted Sentiment'])
    f1_bert = f1_score(comparison_df['Actual Sentiment'], comparison_df['BERT Predicted Sentiment'])

    print("Comparison Metrics:")
    print(f'ELMo Accuracy: {accuracy_elmo:.2f}%')
    print(f'BERT Accuracy: {accuracy_bert:.2f}%')
    print(f'ELMo Precision: {precision_elmo:.2f}')
    print(f'BERT Precision: {precision_bert:.2f}')
    print(f'ELMo Recall: {recall_elmo:.2f}')
    print(f'BERT Recall: {recall_bert:.2f}')
    print(f'ELMo F1 Score: {f1_elmo:.2f}')
    print(f'BERT F1 Score: {f1_bert:.2f}')

    # Save the comparison dataframe
    comparison_csv_path = "/mnt/c/Users/cleme/OneDrive/Desktop/Bachelorarbeit Sentiment/Code/comparison_sentiment_predictions.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Comparison CSV saved to: {comparison_csv_path}")

    return comparison_df

def main():
    elmo_csv_path = "/mnt/c/Users/cleme/OneDrive/Desktop/Bachelorarbeit Sentiment/Code/elmo_sentiment_predictions.csv"
    bert_csv_path = "/mnt/c/Users/cleme/OneDrive/Desktop/Bachelorarbeit Sentiment/Code/bert_sentiment_predictions.csv"

    elmo_df = pd.read_csv(elmo_csv_path)
    bert_df = pd.read_csv(bert_csv_path)

    comparison_df = compare_predictions(elmo_df, bert_df)

    # Plot confusion matrix for ELMo
    plot_confusion_matrix(comparison_df['Actual Sentiment'], comparison_df['ELMo Predicted Sentiment'], "ELMo")

    # Plot confusion matrix for BERT
    plot_confusion_matrix(comparison_df['Actual Sentiment'], comparison_df['BERT Predicted Sentiment'], "BERT")

    # Plot ROC curve for ELMo
    plot_roc_curve(comparison_df['Actual Sentiment'], elmo_pred_proba, "ELMo")

    # Plot ROC curve for BERT
    plot_roc_curve(comparison_df['Actual Sentiment'], bert_pred_proba, "BERT")

if __name__ == "__main__":
    main()