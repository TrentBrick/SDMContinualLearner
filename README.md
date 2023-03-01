# Sparse Distributed Memory is a Continual Learner

This is the codebase behind the paper *Sparse Distributed Memory is a Continual Learner*

Link to paper: https://openreview.net/forum?id=JknGeelZJpHP

Authors: Trenton Bricken, Xander Davies, Deepak Singh, Dmitry Krotov, Gabriel Kreiman

We provide code sufficient to reproduce all of our experiments (aside from FashionMNIST).

Follow these steps:

## 1. Set up a new Conda environment using:

```
conda create --name SDMContLearn python=3.9
conda activate SDMContLearn
conda install pip
pip install setuptools==59.5.0
```

## 2. Clone this github repo and install its requirements:

```
git clone https://github.com/anon8371/AnonPaper1.git
cd AnonPaper1
pip install -r requirements.txt
```

You may be able to use other versions of the libraries found in `requirements.txt` but no promises.

## 3. Install Numenta's Active Dendrites codebase (this is only for one of the benchmarks. Otherwise comment out all imports to it):

```
cd nta/
git clone https://github.com/numenta/htmpapers
cd htmpapers/biorxiv/going_beyond_the_point_neuron
pip install -r requirements.txt
git checkout c59a339a05478f5ae76fc6243c553ef125c0c51a
cd ../../../../
```

## 4. Set up the datasets:

```
cd py_scripts/
python setup_cont_learning_datasets.py
cd ..
```

This downloads MNIST and CIFAR into the correct directories and splits them into 5 disjoint sets for continual learning.

We already provide ConvMixer embeddings of CIFAR10 that were used in `ConvMixerEmbeddings/CIFAR10`. These are copied into the `data/` directory and then also split using the naive ordinal split and then 4 other random splits.

The ImageNet32 embeddings are available for download [here](https://drive.google.com/drive/folders/1jmpuQ7o6wWMllfilo-Fh1Q1_szOludMD?usp=sharing) (they are too large for GitHub). Put them inside `data/ConvMixerEmbeddings/ImageNet32/` to be useable.

If you want the raw ImageNet32 pixels you'll need to get approval to download them [here](https://image-net.org/download.php). You will then need to put them into a directory at `data/ImageNet32/` and then run `ImageNet32_torchify.py` inside `py_scripts/`.


## 5. Setup Wandb

Make an account [here](https://wandb.ai/home) and put the details into `test_runner.py` under `wandb_logger=`.

Otherwise, currently only the validation accuracy for each epoch will be printed out during training.

## 6. Run `python test_runner.py` that will by default run an SDM model on Split MNIST.

See `exp_commands/` for all parameters used to run all of our experiments.

You can put these parameters into `test_runner.py` to run them and fill in `load_path=` with a trained model. To reproduce all of our results we recommend using a job parallelizer like Ray or SLURM to run each experiment as a different job.

If you want your model runs to be saved change `checkpoint_callback = False` on line 57 to `True`. Otherwise, only the continual learning models make while investigating continual learning will be saved out.

See the NISPA and FlyModel folders for the code from these papers that we made compatible with our datasets (`TestingFlyModel.ipynb`). For NISPA, the datasets need to be provided using `cp data/ NISPA/NISPA/Data` or redirecting the datapath inside NISPA.

# Code Base Overview

* **Folders**:
    * data/ - folder containing all datasets
        * data/splits/ - all split versions of the data
        * data/ConvMixerEmbeddings/ - all ConvMixer embedded image data.
    * py_scripts/ - all supporting scripts for training the models.
    * models/ - contains all of our model architecture code
    * exp_commands/ - all major experiments that were run with their relevant hyperparameters
    * notebooks/ - jupyter notebooks used for making a number of the figures and investigating continual learning. Other figures were made using WandB.


* `py_scripts/`:
    - **Data preprocessing**
        - setup_cont_learning_datasets.py - *Calls many of the following scripts to download the MNIST and CIFAR datasets, turn them into PyTorch tensors, and split them for continual learning.*
        - CIFAR10_torchify.py
        - CIFAR100_torchify.py
        - ImageNet32_torchify.py
        - CIFAR10_split_datasets.py
        - MNIST_split_datasets.py
        - CIFAR100_split_dataset.py
        - CIFAR10_cached_data_split.py
        - CIFAR10_cached_data_split_randomize.py
        - cached_data_split_randomize.py

    - **Jupyter Notebook Functions**
        - analyze_training.py
        - utils_SDM_to_Top_k.py

    - **Model and Dataset Parameters**
        - model_params.py - sets all default model parameters
        - dataset_params.py - defines dataset parameters
        - combine_params.py - combines model and dataset parameters from: those set by the experiment, default settings, and those of pretrained models now loaded in

    - data_loaders.py - uses the parameters set above to create dataloaders for pytorch lightning.

    - **Active Dendrites Preprocessing**
        - compute_context_vector.py - general computing of context vectors for Active Dendrites
        - compute_context_vector_MNIST.py - computes context vectors for MNIST specifically. Run as part of `setup_cont_learning_datasets.py`

    - **Helper Scripts to Download ImageNet32**
        - srun_download_imagenet.sh - useful in a Slurm environment
        - srun_untar_imagenet.sh

* `exp_commands/`:
    - **Pretraining**
        - pretrain_on_ImgNet32_embeddings.py - pretrain models for continual learning. Uses the ConvMixer embeddings
        - pretrain_on_ImgNet32_pixels.py - pretraining but on ImageNet32 pixels directly.

    - **Continual Learning**
        - cont_learn_on_SplitCIFAR10_embeddings.py - evaluation used in the main text. Fig 2 and Table 1.
        - cont_learn_on_SplitCIFAR10_embeddings_no_pretrain.py - No ImageNet32 pretraining
        - cont_learn_on_SplitCIFAR10_pixels.py - raw CIFAR10 pixels, no embeddings.
        - cont_learn_on_SplitMNIST.py - on MNIST pixels. No pretraining.
        - cont_learn_on_CIFAR100_embeddings.py - need to set up CIFAR100 by downloading and then torchifying it for this to work.

    - **Ablations** - varying different model components.
        - ablate_test_sdm_optimizers.py - how stale momentum optimizers affect dead neurons and SDM during continual learning.
        - ablate_gaba_switch_timing.py - how soon can the GABA switch start and still give manifold tiling?
        - ablate_pretrain_k_vals_10K_Neurons_ImgNet32_embeddings.py - SDM different k values performance during ImgNet pretraining with 10K neurons.
        - ablate_test_k_vals_10K_Neurons_SplitCIFAR10_embeddings.py - best k values for SplitCIFAR10 continual learning performance with 10K neurons.
        - ablate_pretrain_k_vals_1K_Neurons_ImgNet32_embeddings.py - same as above but with 1K neurons
        - ablate_test_k_vals_1K_Neurons_SplitCIFAR10_embeddings.py - same as above but with 1K neurons

    - **Investigations** - looking at SDM learning dynamics and additional model ablations
        - investigate_cont_learning.py - saving out how many times each neuron is activated and checkpointing the model every 100 epochs as it trains.
        - investigate_deadneuron_GABAswitch_vs_subtract.py.py - how k annealing with subtraction performs vs the GABA switch for solving the dead neuron problem  (it only has an effect when neuron activation values can be negative.)
        - investigate_and_log_CIFAR10_stale_gradients.py - logging the gradients produced by different optimizers during training on CIFAR10

    - **Extras**
        - oracle_train.py - fitting ReLU models on the full dataset to be able to compute maximum performance
        - get_nice_receptive_fields_10K.py - training SDM on CIFAR10 pixels with 10K neurons to learn nice receptive fields.

    - **ConvSDM Models** - Joint training of SDM module with ConvMixer
        - pretrain_ConvSDM_on_ImageNet_pixels.py
        - pretrain_ConvSDM_1K_Neurons_on_CIFAR10_pixels.py
        - pretrain_ConvSDM_10K_Neurons_on_CIFAR10_pixels.py
        - cont_learn_ConvSDM_SplitCIFAR10_pixels.py


* `notebooks/`:
    - InvestigatingContinualLearningWeights.ipynb - broad range of metrics as each model undergoes continual learning
    - Targeted_InvestigatingContinualLearning.ipynb - more targeted plots investigating the continual learning dynamics.
    - StaleGradients.ipynb - toy model implementing the different optimizers and their stale momentums.
    - ReLUStillExponentialDecay.ipynb - analyzing that the new weighted SDM neuron activations still are approximately exponential giving the approximation to Transformer Attention.
    - kValueFrontier.ipynb - summarizing the k value ablations for pretraining and continual learning

    - **Neurogenesis paper replication** - replicating the code of ["A functional model of adult dentate gyrus neurogenesis"](https://www.semanticscholar.org/paper/A-functional-model-of-adult-dentate-gyrus-Gozel-Gerstner/cb5c8122b52062b4d69504f919c6ffdaf0abc965)
        - MATLAB_Neurogenesis_Prep.ipynb
     testing_metrics_notebook.ipynb - preprocessing the MNIST digits as per the paper
        - Neurogenesis.ipynb - relicating the codebase in pytorch and analyzing how well the neurons approximate Top-K.
