import argparse
import Preprocessing
import tism
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from allennlp.modules.elmo import Elmo, batch_to_ids
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

class ElmoClassifier(nn.Module):
    def __init__(self):
        super(ElmoClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, elmo_embeds):
        elmo_embeds = elmo_embeds.mean(dim=1)
        x = self.dropout(nn.functional.relu(self.fc1(elmo_embeds)))
        x = self.dropout(nn.functional.relu(self.fc2(x)))
        logits = self.fc3(x)
        return logits

def pad_collate_fn(batch):
    max_length = max(x['elmo_embeds'].size(0) for x in batch)
    elmo_embeds_padded = []
    labels = []
    for x in batch:
        padding_length = max_length - x['elmo_embeds'].size(0)
        if padding_length > 0:
            padding = torch.zeros((padding_length, x['elmo_embeds'].size(1)), device=x['elmo_embeds'].device)
            padded_embeds = torch.cat((x['elmo_embeds'], padding), dim=0)
        else:
            padded_embeds = x['elmo_embeds']
        elmo_embeds_padded.append(padded_embeds)
        labels.append(x['label'])
    return {
        'elmo_embeds': torch.stack(elmo_embeds_padded),
        'label': torch.stack(labels)
    }

def main():
    parser = argparse.ArgumentParser(description="ELMo Sentiment Analysis")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the data to include in the test split")
    parser.add_argument('--subset_size', type=int, help="Subset size for training")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--use_subset', action='store_true', help="Use a subset of the data for quicker testing")
    args = parser.parse_args()

    base_output_dir = "/mnt/c/Users/cleme/OneDrive/Desktop/Datenanalyse"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "elmo_log.txt")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.cuda.empty_cache()

    training_times = []
    memory_usages = []

    log_memory_usage("Before Training Loop", log_file)

    # Load the ELMo model
    options_file = "/mnt/c/Users/cleme/Downloads/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "/mnt/c/Users/cleme/Downloads/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0).to(device)

    class ElmoDataset(Dataset):
        def __init__(self, texts, labels, elmo):
            self.texts = texts
            self.labels = labels
            self.elmo = elmo

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            character_ids = batch_to_ids([text.split()]).to(device)
            if character_ids.size(1) == 0:
                character_ids = batch_to_ids([[""]]).to(device)
            elmo_embeds = self.elmo(character_ids)['elmo_representations'][0]
            elmo_embeds = elmo_embeds.squeeze(0)
            return {
                'elmo_embeds': elmo_embeds,
                'label': torch.tensor(label, dtype=torch.long)
            }

    train_dataset_elmo = ElmoDataset(X_train, y_train, elmo)
    test_dataset_elmo = ElmoDataset(X_test, y_test, elmo)

    if args.subset_size:
        train_dataset_elmo = torch.utils.data.Subset(train_dataset_elmo, list(range(args.subset_size)))

    train_loader_elmo = DataLoader(train_dataset_elmo, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    test_loader_elmo = DataLoader(test_dataset_elmo, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)

    elmo_model_torch = ElmoClassifier().to(device)
    optimizer_elmo = optim.Adam(elmo_model_torch.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    elmo_model_torch.train()
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(tqdm(train_loader_elmo, desc="Training")):
            optimizer_elmo.zero_grad()
            elmo_embeds = batch['elmo_embeds'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = elmo_model_torch(elmo_embeds)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer_elmo)
            scaler.update()

            log_memory_usage(f"Epoch {epoch+1} Batch {batch_idx}", log_file)

    log_memory_usage("After Training Loop", log_file)

    training_time_elmo = time.time() - start_time
    training_times.append(training_time_elmo)

    elmo_model_torch.eval()
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    all_reviews = X_test.tolist()

    log_memory_usage("Before Evaluation", log_file)

    with torch.no_grad():
        for batch in tqdm(test_loader_elmo, desc="Evaluating"):
            elmo_embeds = batch['elmo_embeds'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = elmo_model_torch(elmo_embeds)
            
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    log_memory_usage("After Evaluation", log_file)

    process = psutil.Process()
    memory_usages.append(process.memory_info().rss / (1024 * 1024))

    df_elmo = pd.DataFrame({
        'Review': all_reviews,
        'Actual Sentiment': all_labels,
        'Predicted Sentiment': all_preds,
        'Predicted Probability': all_probs
    })
    df_elmo.to_csv(os.path.join(output_dir, 'elmo_sentiment_predictions.csv'), index=False)

    accuracy_elmo = 100 * correct / total
    precision_elmo = precision_score(all_labels, all_preds)
    recall_elmo = recall_score(all_labels, all_preds)
    f1_elmo = f1_score(all_labels, all_preds)
    
    summary = (
        f'ELMo Test Accuracy: {accuracy_elmo:.2f} %\n'
        f'Precision: {precision_elmo:.2f}\n'
        f'Recall: {recall_elmo:.2f}\n'
        f'F1 Score: {f1_elmo:.2f}\n'
        f'Training Time: {training_time_elmo:.2f} seconds\n'
        f"\nSummary of ELMo Metrics:\n"
        f"Training Time: {training_times[-1]:.2f} seconds\n"
        f"Memory Usage: {memory_usages[-1]:.2f} MB\n"
        f"Test Accuracy: {accuracy_elmo:.2f} %\n"
        f"Precision: {precision_elmo:.2f}\n"
        f"Recall: {recall_elmo:.2f}\n"
        f"F1 Score: {f1_elmo:.2f}\n"
    )
    print(summary)
    
    with open(log_file, "a") as f:
        f.write(summary)

if __name__ == "__main__":
    main()
