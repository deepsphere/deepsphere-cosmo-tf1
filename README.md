# Spherical convolutional neural network

The code in this repository implements an efficient generalization of the
popular Convolutional Neural Networks (CNNs) to a sphere.

## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/sdsc/scnn
   cd scnn
   ```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
   ```sh
   pip install -r requirements.txt
   ```

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

[usage]: https://github.com/SwissDataScienceCenter/scnn/demo.ipynb
