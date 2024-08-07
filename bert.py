import argparse
import Preprocessing
import tism
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from tqdm import tqdm
import pandas as pd
import psutil
import GPUtil
import os
from datetime import datetime

def log_memory_usage(step="", log_file=None):
    process = psutil.Process()
    memory_info = f"{step} - Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB"
    gpu_memory_info = f"{step} - GPU Memory Usage: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB" if torch.cuda.is_available() else "No GPU available"
    gpus = GPUtil.getGPUs()
    gpu_usage_info = f"{step} - GPU Usage: {gpus[0].load * 100 if gpus else 'N/A'} %"

    log_message = f"{memory_info}\n{gpu_memory_info}\n{gpu_usage_info}\n"
    print(log_message)
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_message + "\n")

def main():
    parser = argparse.ArgumentParser(description="BERT Sentiment Analysis")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the data to include in the test split")
    parser.add_argument('--subset_size', type=int, help="Subset size for training")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--use_subset', action='store_true', help="Use a subset of the data for quicker testing")
    args = parser.parse_args()

    base_output_dir = "/mnt/c/Users/cleme/OneDrive/Desktop/Datenanalyse"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "bert_log.txt")

    print("Loading configuration settings...")
    config_dic = tism.read_config("settings.ini")
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = Preprocessing.preprocessing(config_dic, test_size=args.test_size)

    if args.use_subset:
        train_subset_size = min(args.subset_size, len(X_train))
        test_subset_size = min(args.subset_size, len(X_test))
        X_train = X_train[:train_subset_size]
        y_train = y_train[:train_subset_size]
        X_test = X_test[:test_subset_size]
        y_test = y_test[:test_subset_size]
    print(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")

    class ReviewDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_len,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }

    print("Initializing BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)

    print("Creating datasets and dataloaders...")
    train_dataset = ReviewDataset(X_train, y_train, tokenizer, max_len=128)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer, max_len=128)

    if args.subset_size:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(args.subset_size)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    training_times = []
    memory_usages = []

    log_memory_usage("Before Training Loop", log_file)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    model.train()
    for epoch in range(3):
        print(f'Epoch {epoch + 1}/3')
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            log_memory_usage(f"Epoch {epoch+1} Batch {batch_idx}", log_file)

    log_memory_usage("After Training Loop", log_file)

    training_time = time.time() - start_time
    training_times.append(training_time)

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    all_reviews = X_test.tolist()

    log_memory_usage("Before Evaluation", log_file)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
            
            _, predicted = torch.max(outputs.logits, 1)
            probabilities = torch.softmax(outputs.logits, dim=1)[:, 1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    log_memory_usage("After Evaluation", log_file)

    process = psutil.Process()
    memory_usages.append(process.memory_info().rss / (1024 * 1024))

    df_bert = pd.DataFrame({
        'Review': all_reviews,
        'Actual Sentiment': all_labels,
        'Predicted Sentiment': all_preds,
        'Predicted Probability': all_probs
    })
    df_bert.to_csv(os.path.join(output_dir, 'bert_sentiment_predictions.csv'), index=False)

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    summary = (
        f'BERT Test Accuracy: {accuracy:.2f} %\n'
        f'Precision: {precision:.2f}\n'
        f'Recall: {recall:.2f}\n'
        f'F1 Score: {f1:.2f}\n'
        f'Training Time: {training_time:.2f} seconds\n'
        f"\nSummary of BERT Metrics:\n"
        f"Training Time: {training_times[-1]:.2f} seconds\n"
        f"Memory Usage: {memory_usages[-1]:.2f} MB\n"
        f"Test Accuracy: {accuracy:.2f} %\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1 Score: {f1:.2f}\n"
    )
    print(summary)
    
    with open(log_file, "a") as f:
        f.write(summary)

if __name__ == "__main__":
    main()
