## ECNet

ECNet: Spin-aware electronic-configuration encoding for composition-based prediction of the thermodynamic stability of inorganic compounds


## Installation

```shell
git clone https://github.com/LeiJin-by/ECNet
cd ECNet
conda create -n ECNet python=3.8.0
conda activate ECNet
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install pymatgen matminer
pip install -r requirements.txt

```

## Usage

You can train a model by:

```shell
python train.py --name MP_test --path data/datasets/MP_all.csv --epochs 10
```
After training the model, you can predict using composition
```shell
python predict.py --name MP_test --path data/datasets/MP_all.csv
```

#### Train under different data size

You can set the --train_data_used parameter to specify the proportion of the training set to use.
```shell
python train.py --name customized_model_name --path data/datasets/MP_all.csv --train_data_used 0.5
```
Please type `python train.py --h` for more help.
```shell
optional arguments:
  --path                Path to the dataset.
  --epochs              Number of epochs to train the model.
  --batchsize           Batch size for training.
  --train               whether to train the model.
  --name                Name of the experiment or model
  --train_data_used     Fraction of training data to be used
  --device              Device to run the training on, e.g., 'cuda:0' or 'cpu'.
  --folds               Number of folds for training ECNet
  --lr                  Learning rate for the optimizer.
  --save_model          Whether to save trained models , 1: true, 0: false, default: 1
  --prediction_model    Train a model for predicting or testing , 1: true, 0: false, default: 0
  --performance_test    Whether to test the performance of trained model , 1: true, 0: false, default: 1
```


