This thesis implements three effective techniques for tuning hyperparameters in five diverse
machine learning and deep learning models. The three techniques used are:
• Random search-based tuning.
• Hybrid random and grid search-based tuning.
• Hybrid random and manual search-based tuning.
These techniques are applied to models such as PointNet CNN, regularized and non-regularized
standard feedforward neural networks (FNNs), support vector machine (SVM), and principal
component analysis (PCA). The random search-based tuning involves performing coarse and
fine-tuning stages to optimize the hyperparameters. The hybrid random and grid search combines
the benefits of random search and grid search to find optimal hyperparameter values efficiently.
The hybrid random and manual search utilizes the insights gained from random search as prior
knowledge to guide the manual search process.
These techniques aim to enhance the visual classification accuracy of convolutional deep

neural networks, specifically PointNet. The hyperparameters targeted for tuning are the mini-
batch size of stochastic gradient descent (SGD), momentum, and learning rate. The objective is

to prevent overfitting, improve generalization, and achieve low variance in the model's
performance. The evaluation of the tuned models was conducted using pool and test datasets,
ensuring the models' ability to accurately classify unseen data from different distributions.

Overall, these hyperparameter tuning techniques aimed to optimize the performance and
generalization capabilities of the machine learning and deep learning models, specifically in the
context of visual classification tasks. Furthermore, the three tuning techniques successfully
identified a model for a standard feedforward neural network (FNN) that achieved a high binary
classification accuracy by tuning the regularization parameter (λ), learning rate (α), and
momentum (β). Additionally, the three hyperparameter tuning techniques were effective in
enabling the SVM model to achieve a high binary classification by the regularization parameter
(C) and Gaussian radial basis function (RBF) kernel’s decay coefficient (σ). Moreover, these
techniques successfully determined the minimum number of PCA components that reduced the
dimensionality of the human faces dataset in a computer vision application, keeping possession
of most of its original variance before reduction. Overall, the hyperparameter tuning techniques
showcased their ability to enhance the performance and accuracy of various models across
different datasets and tasks.

##Contribution*
The work described in this thesis strives to improve the learning model’s ability to
generalize to an unseen dataset, either the cross-validation or test dataset, besides the training
dataset. Random search, hybrid random and grid search, and hybrid random and manual
search tuning techniques were utilized to optimize the cost function and generalization of
some machine and deep learning models such as PointNet CNN, regularized and non-
regularized standard feedforward neural networks (FNNs), support vector machine (SVM),
and principal component analysis (PCA).

To the best knowledge of the thesis author that merging random search with grid search or
with manual search has not been reported for tuning the hyperparameters of PointNet CNN,
FNN, SVM, and PCA in the literature of machine and deep learning.
The main contributions of the thesis can be summarized in the following points:
• Using the hybrid random and grid search to optimize the grid search by taking
advantage of random search to determine the efficient hyperparameter space. This
optimized grid search avoids wasted time in tuning the hyperparameters with
inefficient values that would never optimize the learning model’s accuracy.
• Using the hybrid random and manual search to optimize the manual search by
taking advantage of random search at having prior knowledge about the
productive hyperparameter space. This optimized manual search avoids wasted
time at tuning the hyperparameters with inefficient values that would never
optimize the learning model’s accuracy.
• Tuning the hyperparameters of PointNet CNN through three stages is
unprecedented. Particularly, coarse and fine random search or hybrid random and
grid search, or hybrid random and manual search are conducted using a cross-
validation dataset. In addition, a test dataset-based evaluation was performed to
ensure PointNet’s capability to generalize to cross-validation and testing datasets.
Accordingly, the learning model has a high potential to generalize to other unseen
datasets, such as the application dataset.
• Proposing model-independent tuning techniques that can be applied to all deep
CNN networks and machine learning models.
• Obtaining high classification accuracy by applying the three tuning techniques on
standard FNN and SVM, outperforming the state-of-the-art work that uses the
same dataset (make_moons of Sklearn library).
• Applying the three tuning techniques on a supervised learning model (PCA) for
the sake of dimensionality reduction of image datasets.
• Obtaining high classification accuracy without the high memory usage and long
computation time required by the other iterative hyperparameter tuning
techniques like Bayesian, gradient descent, and population-based training
techniques. The iterative method is effective when the CNN model has a high-
dimensional hyperparameter space. This thesis work tunes only at most three
hyperparameters.

## PointNet: *Deep Learning on Point Sets for 3D Classification and Segmentation*
Created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="http://ai.stanford.edu/~haosu/" target="_blank">Hao Su</a>, <a href="http://cs.stanford.edu/~kaichun/" target="_blank">Kaichun Mo</a>, <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from Stanford University.

![prediction example](https://github.com/charlesq34/pointnet/blob/master/doc/teaser.png)

### Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/1612.00593), which is going to appear in CVPR 2017. We proposed a novel deep net architecture for point clouds (as unordered point sets). You can also check our [project webpage](http://stanford.edu/~rqi/pointnet) for a deeper introduction.

Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images. This, however, renders data unnecessarily voluminous and causes issues. In this paper, we design a novel type of neural network that directly consumes point clouds, which well respects the permutation invariance of points in the input.  Our network, named PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective.

In this repository, we release code and data for training a PointNet classification network on point clouds sampled from 3D shapes, as well as for training a part segmentation network on ShapeNet Part dataset.

### Citation
If you find our work useful in your research, please consider citing:

	@article{qi2016pointnet,
	  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
	  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
	  journal={arXiv preprint arXiv:1612.00593},
	  year={2016}
	}
   
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. You may also need to install h5py. The code has been tested with Python 2.7, TensorFlow 1.0.1, CUDA 8.0 and cuDNN 5.1 on Ubuntu 14.04.

If you are using PyTorch, you can find a third-party pytorch implementation <a href="https://github.com/fxia22/pointnet.pytorch" target="_blank">here</a>.

To install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

### Usage
To train a model to classify point clouds sampled from 3D shapes:

    python train.py

Log files and network parameters will be saved to `log` folder in default. Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.

To see HELP for the training script:

    python train.py -h

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir log

After the above training, we can evaluate the model and output some visualizations of the error cases.

    python evaluate.py --visu

Point clouds that are wrongly classified will be saved to `dump` folder in default. We visualize the point cloud by rendering it into three-view images.

If you'd like to prepare your own data, you can refer to some helper functions in `utils/data_prep_util.py` for saving and loading HDF5 files.

### Part Segmentation
To train a model for object part segmentation, firstly download the data:

    cd part_seg
    sh download_data.sh

The downloading script will download <a href="http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html" target="_blank">ShapeNetPart</a> dataset (around 1.08GB) and our prepared HDF5 files (around 346MB).

Then you can run `train.py` and `test.py` in the `part_seg` folder for training and testing (computing mIoU for evaluation).

### License
Our code is released under MIT License (see LICENSE file for details).

### Selected Projects that Use PointNet

* <a href="http://stanford.edu/~rqi/pointnet2/" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017) A hierarchical feature learning framework on point clouds. The PointNet++ architecture applies PointNet recursively on a nested partitioning of the input point set. It also proposes novel layers for point clouds with non-uniform densities.
* <a href="http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w13/Engelmann_Exploring_Spatial_Context_ICCV_2017_paper.pdf" target="_blank">Exploring Spatial Context for 3D Semantic Segmentation of Point Clouds</a> by Engelmann et al. (ICCV 2017 workshop). This work extends PointNet for large-scale scene segmentation.
* <a href="https://arxiv.org/abs/1710.04954" target="_blank">PCPNET: Learning Local Shape Properties from Raw Point Clouds</a> by Guerrero et al. (arXiv). The work adapts PointNet for local geometric properties (e.g. normal and curvature) estimation in noisy point clouds.
* <a href="https://arxiv.org/abs/1711.06396" target="_blank">VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection</a> by Zhou et al. from Apple (arXiv) This work studies 3D object detection using LiDAR point clouds. It splits space into voxels, use PointNet to learn local voxel features and then use 3D CNN for region proposal, object classification and 3D bounding box estimation.
* <a href="https://arxiv.org/abs/1711.08488" target="_blank">Frustum PointNets for 3D Object Detection from RGB-D Data</a> by Qi et al. (arXiv) A novel framework for 3D object detection with RGB-D data. The method proposed has achieved first place on KITTI 3D object detection benchmark on all categories (last checked on 11/30/2017).
