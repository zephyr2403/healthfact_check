import os
import torch
import argparse
import pandas as pd
from transformers import DistilBertTokenizer


def preprocess(dataset_path):
    train = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    validation = pd.read_csv(os.path.join(dataset_path, 'validation.csv'))
    
    train = train[['claim', 'label']].dropna()
    test = test[['claim', 'label']].dropna()
    validation = validation[['claim', 'label']].dropna()
    
    train['claim'] = train['claim'].str.lower()
    test['claim'] = test['claim'].str.lower()
    validation['claim'] = validation['claim'].str.lower()
    return train, test, validation

def tokenize_and_save(data,folder_path,file_name):
    tokenizer = DistilBertTokenizer.from_pretrained('austinmw/distilbert-base-uncased-finetuned-health_facts')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
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
    parser.add_argument('--dataset-path', type=str, default='./dataset', help='Path to the dataset directory')
    parser.add_argument('--preprocess-path', type=str, default='./preprocess', help='Path to save preprocessed tokens')

    args = parser.parse_args()
    
    train, test, validation = preprocess(args.dataset_path)
    tokenize_and_save(train, args.preprocess_path,'train_tokens.pt')
    tokenize_and_save(validation, args.preprocess_path,'validation_tokens.pt')
    tokenize_and_save(test, args.preprocess_path,'test_tokens.pt')

    
