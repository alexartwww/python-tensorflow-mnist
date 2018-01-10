# Python TensorFlow MNIST DATABASE test

This project tests TensorFlow with MNIST DATABASE. For python 2.7

This code based on https://www.tensorflow.org/tutorials/layers tutorial.

# Quickstart

```#!bash

sudo apt-get install python-pip python-dev
sudo pip install tensorflow

git clone https://github.com/alexartwww/python-tensorflow-mnist.git
cd python-tensorflow-mnist

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O source/train-images.idx3-ubyte.gz
gunzip source/train-images.idx3-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O source/train-labels-idx1-ubyte.gz
gunzip source/train-labels-idx1-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O source/t10k-images-idx3-ubyte.gz
gunzip source/t10k-images-idx3-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O source/t10k-labels-idx1-ubyte.gz
gunzip source/t10k-labels-idx1-ubyte.gz

python ./train.py
python ./test.py
python ./predict.py
```

# Project Goals

Best result I've got:

```#!bash

$ python ./predict.py 
Reading labels
Reading images
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_task_type': 'worker', '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fe83907b310>, '_save_checkpoints_steps': None, '_keep_checkpoint_every_n_hours': 10000, '_service': None, '_num_ps_replicas': 0, '_tf_random_seed': None, '_master': '', '_num_worker_replicas': 1, '_task_id': 0, '_log_step_count_steps': 100, '_model_dir': 'net', '_save_summary_steps': 100}
2018-01-10 03:07:57.384388: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
INFO:tensorflow:Restoring parameters from net/model.ckpt-3401
Error rate = 9.20%
```

You awesome!
