# Spherical convolutional neural network

The code in this repository implements an efficient generalization of the
popular Convolutional Neural Networks (CNNs) to a sphere.

The implementation is based on [convolutional neural networks on
graphs][gcnn_paper], as implemented [here][gcnn_code].

[gcnn_paper]: https://arxiv.org/abs/1606.09375
[gcnn_code]: https://github.com/mdeff/cnn_graph/

## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/sdsc/scnn
   cd scnn
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   ```
   If you won't be working with a GPU, comment the `tensorflow-gpu==1.6.0` line
   in `requirements.txt` and uncomment the `tensorflow==1.6.0` line.

3. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```


## Using the model

To use our spherical ConvNet on your data, you need:

1. a data matrix where each row is a sample and each column is a feature,
2. a target vector,

See the [usage notebook][usage] for a simple example with fabricated data.
Please get in touch if you are unsure about applying the model to a different
setting.

[usage]: https://github.com/SwissDataScienceCenter/scnn/blob/master/demo.ipynb

## Cite

Please cite our paper if you use this code in your own work:
```
@article{,
  title={TBD},
  author={Perraudin, Nathanaël},
  journal={Astronomy and Computing},
  year={2018}
}
```
