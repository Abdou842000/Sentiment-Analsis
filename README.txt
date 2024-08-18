
Names of the Students:
- Abderrahim NAMOUH
- Ayoub TARGAOUI


Description of the Implemented Classifier:
The classifier implemented is a custom BERT-based model for sentiment analysis. The model utilizes the pre-trained `bert-base-uncased` version of BERT from Hugging Face's Transformers library. It is designed to classify sentiments into three categories: positive, negative, and neutral.

Input and Feature Representation:
The input to the model is a combination of a sentence and its corresponding aspect category, forming a single text input. This combined text is then tokenized using BERT's tokenizer, converting it into a format suitable for the model (input IDs and attention mask). The maximum sequence length for tokenization is set to 128 tokens, with padding or truncation applied as necessary.

Model Architecture:
The model architecture comprises the following layers:
1. BERT Layer: Utilizes `BertModel.from_pretrained("bert-base-uncased")` to generate embeddings for the input tokens.
2. Fully Connected Layer 1: A linear layer that reduces the dimension from BERT's hidden size to 128.
3. Activation Layer: ReLU activation function.
4. Fully Connected Layer 2: The final linear layer that maps the representation to the three sentiment categories (positive, negative, neutral).

Resources:
We used Pre-trained BERT model (`bert-base-uncased`) and BERT Tokenizer.

The training and evaluation of the sentiment analysis model were performed on a Google Compute Engine backend equipped with Python 3 and GPU support. The resource utilization over the course of the computation was as follows:
	- System RAM: 3.5 GB of 12.7 GB used.
	- GPU RAM: 9.1 GB of 15.0 GB used.
These resources are within the range of the GPU that will be used for evaluation.

It is worth noting that when utilizing the provided tester.py script for training our sentiment analysis model, we encountered a constraint in the number of epochs we could feasibly execute. Due to time-intensive operations, we were limited to training our model for no more than 5 epochs. This restriction was primarily due to the extended duration of each epoch when running through the script's routines.

It is worth noting that separate from the tester.py script, we observed a positive correlation between the number of epochs and model performance. Higher epoch counts, when tested independently, led to improved model accuracy. This suggests that the model benefits from extended training time to better learn and adapt to the training data. However, the time constraints posed by the current script setup limited our ability to leverage this potential for enhanced performance.

Future iterations of model training could consider optimizing the script or the training process to allow for more epochs within a practical timeframe, thereby achieving the observed performance gains associated with longer training.

Accuracy on the Dev Dataset:
The accuracy of the model on the development dataset is 83%.

