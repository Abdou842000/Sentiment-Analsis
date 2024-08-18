import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import pandas as pd
from torch.utils.data import  DataLoader
from typing import List
from sentiment import SentimentAnalysisDataset

class Classifier(nn.Module):
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        num_classes = 3  # Assuming 3 classes for sentiment analysis
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

        train_df = pd.read_csv(train_filename, sep='\t', names=['Polarity', 'Aspect_Category', 'Target_Term', 'Offsets', 'Sentence'])
        dev_df = pd.read_csv(dev_filename, sep='\t', names=['Polarity', 'Aspect_Category', 'Target_Term', 'Offsets', 'Sentence'])
          
        train_dataset = SentimentAnalysisDataset(train_df, self.tokenizer)
        dev_dataset = SentimentAnalysisDataset(dev_df, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

        for epoch in range(1): 
            
            for input_ids, attention_mask, labels in train_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        predictions = []

        data_df = pd.read_csv(data_filename, sep='\t', names=['Polarity', 'Aspect_Category', 'Target_Term', 'Offsets', 'Sentence'])
        dataset = SentimentAnalysisDataset(data_df, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        for input_ids, attention_mask, _ in data_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with torch.no_grad():
                outputs = self(input_ids, attention_mask)
                _, predicted = torch.max(outputs, dim=1)
                predictions.extend(predicted.tolist())

        # Assuming a simple mapping for demonstration purposes
        label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        predicted_labels = [label_map[label] for label in predictions]
        return predicted_labels
    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs["last_hidden_state"][:, 0, :]
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.fc2(x)
        return x 




