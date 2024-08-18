# Sentiment Analysis with Custom BERT-based Classifier

## Students
- Abderrahim NAMOUH
- Ayoub TARGAOUI

## Project Overview
This project implements a custom BERT-based classifier for sentiment analysis. The classifier is designed to predict sentiment for given text into three categories:
- Positive
- Negative
- Neutral

The model is based on the pre-trained `bert-base-uncased` version of BERT from Hugging Face's Transformers library and has been fine-tuned to handle sentiment classification tasks.

## Model Architecture

### Input Representation
The model takes as input a combination of a sentence and its corresponding aspect category, forming a single text string. This text is tokenized using BERT's tokenizer, which generates:
- Input IDs
- Attention Mask

The tokenizer pads or truncates the input text to a maximum length of 128 tokens to ensure consistency.

### Model Structure
The model is composed of the following key layers:
1. **BERT Layer**: Pre-trained BERT model (`BertModel.from_pretrained("bert-base-uncased")`) to generate embeddings for the tokenized input.
2. **Fully Connected Layer 1**: A linear layer that reduces BERT's hidden size down to 128 dimensions.
3. **Activation Layer**: ReLU activation function applied after the first fully connected layer.
4. **Fully Connected Layer 2**: The final linear layer that maps the 128-dimensional vector to three sentiment classes: positive, negative, or neutral.

### Resources Used
- **Pre-trained Model**: `bert-base-uncased`
- **Tokenizer**: BERT Tokenizer from Hugging Face

Training and evaluation were performed on a Google Compute Engine backend with GPU support. The resources utilized during training were:
- **System RAM**: 3.5 GB of 12.7 GB used
- **GPU RAM**: 9.1 GB of 15.0 GB used

## Training and Epoch Limitations
While training the model using the `tester.py` script, a constraint was observed in the number of epochs that could be practically executed. Due to time-intensive operations in the script, we were limited to 5 epochs. However, independent tests have shown a positive correlation between the number of epochs and model accuracy, suggesting that extended training would lead to better performance.

### Future Improvements
Future iterations of this project could consider optimizing the training script or process to allow for longer training times, thereby potentially improving the model's performance through additional epochs.

## Accuracy on Development Dataset
The model achieved an accuracy of **83%** on the development dataset.



