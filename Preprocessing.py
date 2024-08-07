import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocessing(config_dic=None, test_size=0.2):
    print_debug("START", config_dic)
    
    # Load the dataset
    file_path = "/mnt/c/Users/cleme/OneDrive/Desktop/Amazon-Product-Reviews-AmazonProductReview.csv"

    # Verify that the file exists at the specified path
    import os
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file was not found at the specified path: {file_path}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Select relevant columns for analysis
    df = df[['review_body', 'sentiment']]

    # Handle NaN values in the 'review_body' column
    df['review_body'] = df['review_body'].fillna('')

    # Ensure all entries in 'review_body' are strings
    df['review_body'] = df['review_body'].astype(str)

    # Clean the text data
    def clean_text(text):
        text = re.sub(r'<br />', ' ', text)  # Replace HTML line breaks with space
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
        text = text.lower()  # Convert text to lowercase
        return text

    df['review_body'] = df['review_body'].apply(clean_text)

    # Remove empty texts after cleaning
    df = df[df['review_body'].str.strip().astype(bool)]
    
    # Encode the sentiment labels
    le = LabelEncoder()
    df['sentiment'] = le.fit_transform(df['sentiment'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['review_body'], df['sentiment'], test_size=test_size, random_state=42)

    # Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    print_debug("Data preprocessing complete.", config_dic)

    return X_train, X_test, y_train, y_test

def print_debug(msg, config_dic=None):
    if config_dic:
        debug_mode = config_dic['DEFAULT'].get('debug_print', 'False').lower() in ('true', '1', 'yes', 'on') 
        # Print the debug message if debug mode is on
        if debug_mode:
            print(msg)
    
if __name__ == "__main__":
    preprocessing()