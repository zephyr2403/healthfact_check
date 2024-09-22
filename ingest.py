import os
import argparse
import pandas as pd 
from datasets import load_dataset

def download_and_save(folder_path='./dataset'):

    dataset = load_dataset('ImperialCollegeLondon/health_fact',trust_remote_code=True)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    train = pd.DataFrame(dataset['train'])
    test = pd.DataFrame(dataset['test'])
    valiadate = pd.DataFrame(dataset['validation'])
    
    train.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(folder_path, 'test.csv'), index=False)
    valiadate.to_csv(os.path.join(folder_path, 'validation.csv'), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and save the PUBMED dataset.")
    parser.add_argument('--path', type=str, default='./dataset', help='Path to save the dataset (default: ./dataset)')
    
    args = parser.parse_args()

    download_and_save(args.path)