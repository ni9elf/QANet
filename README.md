# QANet in Tensorflow

A Tensorflow implementation of [QANet](https://arxiv.org/abs/1804.09541), an efficient reading comprehension model that is exclusively built upon convolutions and self-attentions. By not using recurrent connections, a 13x speedup in training is achieved while maintaining good accuracy. Notably, the EM accuracy score beats human performance on the SquAD dataset.

![architecture](assets/architecture.png)
