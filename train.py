import os 
import torch
import configparser
import argparse
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments


config = configparser.ConfigParser()

config.read('config.ini')

output_dir = config.get('training', 'output_dir')
num_train_epochs = config.getint('training', 'num_train_epochs')
per_device_train_batch_size = config.getint('training', 'per_device_train_batch_size')
per_device_eval_batch_size = config.getint('training', 'per_device_eval_batch_size')
warmup_steps = config.getint('training', 'warmup_steps')
weight_decay = config.getfloat('training', 'weight_decay')
logging_dir = config.get('training', 'logging_dir')
logging_steps = config.getint('training', 'logging_steps')
eval_strategy = config.get('training', 'eval_strategy')
save_strategy = config.get('training', 'save_strategy')
learning_rate = config.getfloat('training', 'learning_rate')

model_name = config.get('model', 'model_name')

def compute_metrics(pred):

    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro',zero_division=0)
    
    acc = accuracy_score(labels, preds)
    
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

class HealthDataLoader(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def load_data(preprocess_path):
    """
    Loads preprocessed tokenized datasets from specified file paths and 
    returns them as instances of the HealthDataLoader.

    This function loads the train, validation, and test datasets from 
    their respective tokenized files, constructs HealthDataLoader 
    instances, and returns them.

    Args:
        preprocess_path (str): Path to the folder containing the tokenized 
                               data files (train_tokens.pt, validation_tokens.pt).

    Returns:
        tuple: Three HealthDataLoader instances for train, validation, and test datasets.

    """
    train_data = torch.load(os.path.join(preprocess_path, 'train_tokens.pt'),weights_only=True)
    validation_data = torch.load(os.path.join(preprocess_path, 'validation_tokens.pt'),weights_only=True)
    test_data = torch.load(os.path.join('./preprocess', 'test_tokens.pt'),weights_only=True)
    
    train_dataset = HealthDataLoader({'input_ids': train_data['input_ids'], 'attention_mask': train_data['attention_mask']}, train_data['labels'])
    validation_dataset = HealthDataLoader({'input_ids': validation_data['input_ids'], 'attention_mask': validation_data['attention_mask']}, validation_data['labels'])
    test_dataset = HealthDataLoader({'input_ids': test_data['input_ids'], 'attention_mask': test_data['attention_mask']}, test_data['labels'])

    return train_dataset, validation_dataset, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DistilBert Model')
    parser.add_argument('--preprocess-path', type=str, default='./preprocess', help='Path to retrieve preprocessed tokens(default: ./preprocess)')
    parser.add_argument('--store-model', type=str, default='./models', help='Path to save trained weight & tokenizer(default: ./models)')

    args = parser.parse_args()
    model_path = args.store_model
    
    train_dataset, validation_dataset, test_dataset = load_data(args.preprocess_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=True)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=4,ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=output_dir,               
        num_train_epochs=num_train_epochs,               
        per_device_train_batch_size=per_device_train_batch_size,  
        per_device_eval_batch_size=per_device_eval_batch_size,   
        warmup_steps=warmup_steps,                      
        weight_decay=weight_decay,                   
        logging_dir=logging_dir,                  
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,           
        save_strategy=save_strategy,                 
        learning_rate=learning_rate
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics= compute_metrics
    )

    trainer.train()

    test_result = trainer.evaluate(test_dataset)
    table_data = [[key, round(value,3)] for key, value in test_result.items()]

    print(tabulate(table_data, headers=["Test Eval Metric", "Value"], tablefmt="grid"))
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)