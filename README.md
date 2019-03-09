# DeepSphere: a spherical convolutional neural network

[Nathanaël Perraudin][nath], [Michaël Defferrard][mdeff], Tomasz Kacprzak, Raphael Sgier

[nath]: https://perraudin.info
[mdeff]: http://deff.ch

The code in this repository implements a generalization of Convolutional Neural Networks (CNNs) to the sphere.
We here model the discretised sphere as a graph of connected pixels.
The resulting convolution is more efficient (especially when data doesn't span the whole sphere) and mostly equivariant to rotation (small distortions are due to the non-existence of a regular sampling of the sphere).
The pooling strategy exploits a hierarchical pixelisation of the sphere (HEALPix) to analyse the data at multiple scales.
The graph neural network model is based on [ChebNet][gcnn_paper] and its [TensorFlow implementation][gcnn_code].
The performance of DeepSphere is demonstrated on a discrimination problem: the classification of convergence maps into two cosmological model classes.

**blog post**: [DeepSphere: a neural network architecture for spherical data][blog]

**paper**: [DeepSphere: Efficient spherical Convolutional Neural Network with HEALPix sampling for cosmological applications][paper]

[blog]: https://datascience.ch/deepsphere-a-neural-network-architecture-for-spherical-data
[paper]: https://arxiv.org/abs/1810.12186

[gcnn_paper]: https://arxiv.org/abs/1606.09375
[gcnn_code]: https://github.com/mdeff/cnn_graph/

## Installation

[![Binder](https://mybinder.org/badge.svg)][binder_lab]
&nbsp; Click the binder badge to play with the notebooks from your browser without installing anything.

[binder_lab]: https://mybinder.org/v2/gh/SwissDataScienceCenter/DeepSphere/outputs?urlpath=lab

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/SwissDataScienceCenter/DeepSphere.git
   cd DeepSphere
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   ```

   **Note**: if you will be working with a GPU, comment the
   `tensorflow==1.6.0` line in `requirements.txt` and uncomment the
   `tensorflow-gpu==1.6.0` line.

   **Note**: the code has been developed and tested with Python 3.5 and 3.6.
   It should work on Python 2.7 with `requirements_py27.txt`.
   Please send a PR if you encounter an issue.

3. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```

## Notebooks

The below notebooks contain examples and experiments to play with the model.
Look at the first two if you want to use the model with your own data.

1. [Classification of data on the whole sphere.][whole_sphere]
   The easiest example to run if you want to play with the model.
1. [Classification of data from part of the sphere with noise.][part_sphere]
   That is the main experiment carried on in the paper (with one configuration of resolution and noise).
   It requires private data, see below.
1. [Spherical convolution using a graph to represent the sphere.][spherical_convolution]
   Learn what is the convolution on a graph, and how it works on the sphere.
   Useful to learn how the model works.
1. [Comparison of the spherical harmonics with the eigenvectors of the graph Laplacian.][spherical_vs_graph]
   Get a taste of the difference between the graph representation and the analytic sphere.

[whole_sphere]: https://nbviewer.jupyter.org/github/SwissDataScienceCenter/DeepSphere/blob/outputs/demo_whole_sphere.ipynb
[part_sphere]: https://nbviewer.jupyter.org/github/SwissDataScienceCenter/DeepSphere/blob/outputs/demo_part_sphere.ipynb
[spherical_convolution]: https://nbviewer.jupyter.org/github/SwissDataScienceCenter/DeepSphere/blob/outputs/demo_spherical_convolution.ipynb
[spherical_vs_graph]: https://nbviewer.jupyter.org/github/SwissDataScienceCenter/DeepSphere/blob/outputs/demo_spherical_vs_graph.ipynb

## Reproducing the results of the paper

[cosmo_eth]: http://www.cosmology.ethz.ch

Follow the below steps to reproduce the paper's results.
While the instructions are simple, the experiments will take a while.

1. **Get the main dataset**.
   You need to ask the [ETHZ cosmology research group][cosmo_eth] for a copy of the data.
   The simplest option is to request access on [Zenodo](https://zenodo.org/record/1303272).
   You will have to write a description of the project for which the dataset is intended to be used.

2. **Preprocess the dataset**.
   ```
   python data_preprocess.py
   ```

3. **Run the experiments**.
   The first corresponds to the fully convolutional architecture variant of DeepSphere.
   The second corresponds to the classic CNN architecture variant.
   The last two are the baselines: an SVM that classifies histograms and power spectral densities.
   ```
   python experiments_deepsphere.py FCN
   python experiments_deepsphere.py CNN
   python experiments_histogram.py
   python experiments_psd.py
   ```

The results will be saved in the [`results`](results) folder.
Note that results may vary from one run to the other.
You may want to check summaries with tensorboard to verify that training converges.
For some experiments, the network needs a large number of epochs to stabilize.

The `experiments_deepsphere.py` and `experiments_psd.py` scripts can be executed in parallel in a HPC setting.
You can adapt the `launch_euler.py` and `launch_cscs.py` scripts to your particular setting.

Once the results are computed (or using those stored in the repository), you can reproduce the paper's figures with the `figure*` notebooks.
The results will be saved in the [`figures`](figures) folder.

## License & citation

The content of this repository is released under the terms of the [MIT license](LICENCE.txt).
Please cite our [paper] if you use it.

```
@article{,
  title = {DeepSphere: Efficient spherical Convolutional Neural Network with HEALPix sampling for cosmological applications},
  author = {Perraudin, Nathana\"el and Defferrard, Micha\"el and Kacprzak, Tomasz and Sgier, Raphael},
  journal = {arXiv},
  year = {2018},
  url = {https://arxiv.org/abs/1810.12186},
}
```
