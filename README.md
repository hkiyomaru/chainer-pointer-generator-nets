# Chainer implementation of Pointer Generator Networks

Chainer-based implementation of Pointer Generator Networks.

See "[Get To The Point: Summarization with Pointer-Generator Networks](http://aclweb.org/anthology/P/P17/P17-1099.pdf)", Abigail See; Peter J. Liu; Christopher D. Manning, ACL 2017.

This repository is partly derived from [this repository](https://github.com/kiyomaro927/chainer-attention-nmt) and Chainer's official [seq2seq example](https://github.com/chainer/chainer/tree/master/examples/seq2seq).

# Development Environment

* Ubuntu 16.04
* Python 3.5.2
* Chainer 3.1.0
* numpy 1.13.3
* cupy 2.1.0
* nltk
* progressbar
* and their dependencies

# How to Run

```
$ python train.py <path/to/training-source> <path/to/training-target> <path/to/source-vocabulary> <path/to/target-vocabulary> --validation-source <path/to/validation-source> --validation-target <path/to/validation-target> -g <gpu-id>
```
