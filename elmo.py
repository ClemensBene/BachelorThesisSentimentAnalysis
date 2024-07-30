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
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="ELMo Sentiment Analysis")
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

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the ELMo model
    options_file = "/mnt/c/Users/cleme/Downloads/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "/mnt/c/Users/cleme/Downloads/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0).to(device)

    # Define a custom dataset class for ELMo
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
            character_ids = batch_to_ids([text.split()]).to(device)  # Ensure character_ids is on GPU
            
            # Check if character_ids is empty
            if character_ids.size(1) == 0:
                # If empty, return a tensor with a single zero token
                character_ids = batch_to_ids([[""]]).to(device)
            
            elmo_embeds = self.elmo(character_ids)['elmo_representations'][0]
            elmo_embeds = elmo_embeds.squeeze(0)
            return {
                'elmo_embeds': elmo_embeds,
                'label': torch.tensor(label, dtype=torch.long)
            }

    # Padding function for collate
    def pad_collate_fn(batch):
        max_length = max(x['elmo_embeds'].size(0) for x in batch)
        elmo_embeds_padded = []
        labels = []
        for x in batch:
            padding_length = max_length - x['elmo_embeds'].size(0)
            if (padding_length) > 0:
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

    # Create training and testing datasets for ELMo
    train_dataset_elmo = ElmoDataset(X_train, y_train, elmo)
    test_dataset_elmo = ElmoDataset(X_test, y_test, elmo)

    # Create data loaders for batching
    train_loader_elmo = DataLoader(train_dataset_elmo, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
    test_loader_elmo = DataLoader(test_dataset_elmo, batch_size=16, shuffle=False, collate_fn=pad_collate_fn)

    # Define a simple neural network model for ELMo
    class ElmoClassifier(nn.Module):
        def __init__(self):
            super(ElmoClassifier, self).__init__()
            self.fc = nn.Linear(1024, 2)

        def forward(self, elmo_embeds):
            # Mean pooling over the sequence length dimension
            elmo_embeds = elmo_embeds.mean(dim=1)
            logits = self.fc(elmo_embeds)
            return logits

    elmo_model_torch = ElmoClassifier().to(device)
    optimizer_elmo = optim.Adam(elmo_model_torch.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop for ELMo
    start_time = time.time()
    elmo_model_torch.train()
    num_epochs = 3
    for epoch in range(num_epochs):  # Train for 3 epochs
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(tqdm(train_loader_elmo, desc="Training")):
            optimizer_elmo.zero_grad()  # Zero the gradients
            elmo_embeds = batch['elmo_embeds'].to(device)
            labels = batch['label'].to(device)
            outputs = elmo_model_torch(elmo_embeds)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer_elmo.step()  # Update the weights
    training_time_elmo = time.time() - start_time

    # Evaluation for ELMo
    elmo_model_torch.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_reviews = X_test.tolist()
    with torch.no_grad():
        for batch in tqdm(test_loader_elmo, desc="Evaluating"):
            elmo_embeds = batch['elmo_embeds'].to(device)
            labels = batch['label'].to(device)
            outputs = elmo_model_torch(elmo_embeds)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Save predictions to CSV
    df_elmo = pd.DataFrame({
        'Review': all_reviews,
        'Actual Sentiment': all_labels,
        'Predicted Sentiment': all_preds
    })
    df_elmo.to_csv('elmo_sentiment_predictions.csv', index=False)

    # Calculate and print ELMo accuracy
    accuracy_elmo = 100 * correct / total
    precision_elmo = precision_score(all_labels, all_preds)
    recall_elmo = recall_score(all_labels, all_preds)
    f1_elmo = f1_score(all_labels, all_preds)
    print(f'ELMo Test Accuracy: {accuracy_elmo} %')
    print(f'Precision: {precision_elmo}')
    print(f'Recall: {recall_elmo}')
    print(f'F1 Score: {f1_elmo}')
    print(f'Training Time: {training_time_elmo} seconds')

if __name__ == "__main__":
    main()
