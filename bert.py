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

def main():
    parser = argparse.ArgumentParser(description="BERT Sentiment Analysis")
    parser.add_argument('--use_subset', action='store_true', help="Use a subset of the data for quicker testing")
    parser.add_argument('--subset_size', type=int, default=100, help="Total size of the subset to use if --use_subset is specified")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the subset to include in the test split if --use_subset is specified")
    args = parser.parse_args()

    # Load the configuration settings
    print("Loading configuration settings...")
    config_dic = tism.read_config("settings.ini")
    
    # Preprocess the data and get training and testing datasets
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = Preprocessing.preprocessing(config_dic)

    # If using a subset, limit the dataset size for quicker testing
    if args.use_subset:
        subset_size = args.subset_size
        split_index = int(subset_size * (1 - args.test_size))
        X_train = X_train[:split_index]
        y_train = y_train[:split_index]
        X_test = X_test[:subset_size - split_index]
        y_test = y_test[:subset_size - split_index]
    print(f"Training data size after applying subset (if used): {len(X_train)}, Testing data size after applying subset (if used): {len(X_test)}")

    # Define a custom dataset class for BERT
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
            # Tokenize the text using BERT tokenizer
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

    # BERT implementation
    print("Initializing BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
    
    # Create training and testing datasets
    print("Creating datasets and dataloaders...")
    train_dataset = ReviewDataset(X_train, y_train, tokenizer, max_len=128)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer, max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop for BERT
    start_time = time.time()
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        print(f'Epoch {epoch + 1}/3')
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()  # Zero the gradients
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            #if batch_idx % 10 == 0:
                #print(f'Batch {batch_idx} input_ids device: {input_ids.device}, labels device: {labels.device}, loss: {loss.item()}')
    training_time = time.time() - start_time

    # Evaluation for BERT
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_reviews = X_test.tolist()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Save predictions to CSV
    df_bert = pd.DataFrame({
        'Review': all_reviews,
        'Actual Sentiment': all_labels,
        'Predicted Sentiment': all_preds
    })
    df_bert.to_csv('bert_sentiment_predictions.csv', index=False)

    # Calculate and print BERT accuracy
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f'BERT Test Accuracy: {accuracy} %')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Training Time: {training_time} seconds')

if __name__ == "__main__":
    main()