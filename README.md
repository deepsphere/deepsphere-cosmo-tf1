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
   git clone https://github.com/SwissDataScienceCenter/scnn.git
   cd scnn
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   ```

   **Note**: if you won't be working with a GPU, comment the
   `tensorflow-gpu==1.6.0` line in `requirements.txt` and uncomment the
   `tensorflow==1.6.0` line.

   **Note**: The code has been developed and tested with Python 3.5 and 3.6. It
   should work on Python 2.7 with `requirements_py27.txt`. Please send a PR if
   you encounter an issue.

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

## Experiments

Below are some notebooks which contain various experiments:
1. Classification of data on the whole sphere
1. Classification of data from part of the sphere
1. Robustness to noise

The results we reported in the paper are recorded here:
1. TODO: link to notebook
1. TODO: link to notebook

## License & co

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
Please cite our [paper][arXiv] if you use it.

```
@article{,
  title={TBD},
  author={Perraudin, Nathanaël},
  journal={Astronomy and Computing},
  year={2018}
}
```
