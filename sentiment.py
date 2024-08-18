from torch.utils.data import Dataset
import torch

class SentimentAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['Sentence']
        aspect = row['Aspect_Category']
        combined_text = f"{aspect} {text}"
        inputs = self.tokenizer.encode_plus(combined_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        label = label_map[row['Polarity']]
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze(), torch.tensor(label)

    def __len__(self):
        return len(self.dataframe)

