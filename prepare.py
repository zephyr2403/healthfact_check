import os
import torch
import argparse
import pandas as pd
from transformers import DistilBertTokenizer


def preprocess(dataset_path):
    """
    Reads and preprocesses the dataset from the specified folder path. 
    The function performs the following steps:
    
    - Reads the 'train.csv', 'test.csv', and 'validation.csv' files.
    - Drops rows with missing values in the 'claim' and 'label' columns.
    - Converts all text in the 'claim' column to lowercase.
    - Filters out rows where the 'label' is -1.
    
    Args:
        dataset_path (str): Path to the folder containing the dataset CSV files.
    
    Returns:
        tuple: Three pandas DataFrames representing the preprocessed train, test, and validation datasets.
    """
    print(f'Reading Data from {dataset_path}')
    train = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    validation = pd.read_csv(os.path.join(dataset_path, 'validation.csv'))
    print(f'Processing Dataframes...')
    train = train[['claim', 'label']].dropna()
    test = test[['claim', 'label']].dropna()
    validation = validation[['claim', 'label']].dropna()
    
    train['claim'] = train['claim'].str.lower()
    test['claim'] = test['claim'].str.lower()
    validation['claim'] = validation['claim'].str.lower()

    train = train[train['label']!=-1]
    test = test[test['label']!=-1] 
    validation = validation[validation['label']!=-1]
    print(f'Dataframe Processed.')
    return train, test, validation

def tokenize_and_save(data,folder_path,file_name):
    """
    Tokenizes the 'claim' texts from the provided DataFrame using a pretrained 
    DistilBERT tokenizer and saves the tokenized inputs and labels to a file.

    Args:
        data (pd.DataFrame): DataFrame containing 'claim' and 'label' columns.
        folder_path (str): Path to the folder where the tokenized data will be saved.
        file_name (str): Name of the file to save the tokenized data.
    
    Saves:
        A tensor file containing 'input_ids', 'attention_mask', and 'labels' 
        at the specified folder path and file name.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('austinmw/distilbert-base-uncased-finetuned-health_facts',    clean_up_tokenization_spaces=True)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(f'Tokenizing {file_name.split("-")[0]}')
    tokens = tokenizer(
        data['claim'].to_list(),
        max_length=512,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='pt'
    )

    label_tensor = torch.tensor(data['label'].to_list())
    torch.save({
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'labels': label_tensor
    }, os.path.join(folder_path, file_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess and tokenize datasets.')
    parser.add_argument('--dataset-path', type=str, default='./dataset', help='Path to the dataset directory(default: ./dataset)')
    parser.add_argument('--preprocess-path', type=str, default='./preprocess', help='Path to save preprocessed tokens(default: ./preprocess)')

    args = parser.parse_args()
    
    train, test, validation = preprocess(args.dataset_path)
    
    tokenize_and_save(train, args.preprocess_path,'train_tokens.pt')
    tokenize_and_save(validation, args.preprocess_path,'validation_tokens.pt')
    tokenize_and_save(test, args.preprocess_path,'test_tokens.pt')
    print('Done.')

    
