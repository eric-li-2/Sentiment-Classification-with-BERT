This repository is a small project analyzing the effect of pre-training BERT for downstream tasks. 
The task I chose was sentiment analysis on SST2 dataset.
In this project, I trained two BERT models, one from scratch directly on SST2, and one pretrained then fine tuned on SST2.

Results:
The fine-tuned model achieves 92.2% accuracy within one epoch.
The from scratch model achieves an 80% accuracy after 5 epochs.
These results indicate that pretraining is important for transformer models. Transformer models have weak inductive bias and thus perform much better with more data.
The model trained from scratch plateaud around 5 epochs and then started overfitting on later epochs.

Run instructions:
1) download pytorch and huggingface datasets
2) run train.py to train a BERT model from random initialization
3) run demo.py for an interactive demo
4) run the python notebook at https://colab.research.google.com/drive/187NLBJrTxKZjpe40H5wIIkyVVDHLmgVJ?usp=sharing for the fine-tuned version

Other details:
This code was run on an aws ec2 g5.2xlarge instance
