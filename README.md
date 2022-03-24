# Transformer - Attention is all you need - Pytorch Implementation

This is a PyTorch implementation of the Transformer model in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017).

In this implementation, I will train the model on the machine translation task from English to Vietnamese, with the data used [here](https://drive.google.com/file/d/1Fuo_ALIFKlUvOPbK5rUA5OfAS2wKn_95/view) ([This is](https://github.com/pbcquoc/transformer) the original reference repository)

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).


<p align="center">
<img src="https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png" width="350">
</p>


The project support training and translation with trained model now.

If there is any suggestion or error, feel free to fire an issue to let me know. :)


# Usage

## English-Vietnamese Translation: en-vi

### 1) Download the dataset
```bash
gdown --id 1Ty1bGrd0sCwEqXhsoViCUaNKa3lFwmPH
unzip en_vi.zip
rm en_vi.zip
mv data/ data_en_vi/
```

### 2) Train the model
```bash
python train.py
```

### 3) Test the model
```bash
python translate.py
```
---
# Performance

- Parameter settings:
  - batch size 40 
  - epoch 50 
  - learning_rate 0.0001
  - cross_entropy loss
 
  
## Testing 
- Soon.
---
# TODO
  - Evaluation on the generated text.
  - Attention weight plot.
---
# References
