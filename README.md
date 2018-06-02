# QANet in Tensorflow

A Tensorflow implementation of [QANet](https://arxiv.org/abs/1804.09541), an efficient reading comprehension model that is exclusively built upon convolutions and self-attentions. By not using recurrent connections, a 13x speedup in training is achieved while maintaining good accuracy. Notably, the EM accuracy score beats human performance on the SQuAD dataset.

<p align="center"> 
<img src="assets/architecture.png">
</p>

## Requirements

* Python 2.7
* Tensorflow 1.8
* NumPy 1.14.2

## Code Organization

The OOP paradigm is followed in structuring the model code in classes. Modularity through extensive use of functions helps abstract complex architecture parts and even Tensorflow functionality (ex: sharing of tensors).

## Author

[@ni9elf](https://github.com/ni9elf)

Nigel Fernandez 

