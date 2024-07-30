import pandas as pd

def inspect_csv(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.columns)

elmo_path = "/mnt/c/Users/cleme/OneDrive/Desktop/Bachelorarbeit Sentiment/Code/elmo_sentiment_predictions.csv"
bert_path = "/mnt/c/Users/cleme/OneDrive/Desktop/Bachelorarbeit Sentiment/Code/bert_sentiment_predictions.csv"

print("Inspecting ELMo CSV:")
inspect_csv(elmo_path)

print("\nInspecting BERT CSV:")
inspect_csv(bert_path)
