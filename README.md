# ECNet

ECNet: Spin-aware electronic-configuration encoding for composition-based prediction of the thermodynamic stability of inorganic compounds.

ECNet is a composition-based model that uses only chemical formulas as input. It combines element identity, binary stoichiometric count encoding, and a compact 137-dimensional electronic-configuration descriptor for thermodynamic-stability prediction.

## Installation

```shell
git clone https://github.com/LeiJin-by/ECNet
cd ECNet

conda create -n ECNet python=3.8.0
conda activate ECNet

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

If a different CUDA version is used, please install the corresponding PyTorch version following the official PyTorch instructions.

## Usage

Train ECNet on a benchmark dataset:

```shell
python train.py --name ECNet_MP --path data/datasets/MP_all.csv --folds 5 --epochs 100 --device cuda:0
```

Predict with trained models:

```shell
python predict.py --name ECNet_MP --path data/datasets/MP_all.csv --device cuda:0
```

Train with a specified fraction of the training data:

```shell
python train.py --name ECNet_MP_50percent --path data/datasets/MP_all.csv --train_data_used 0.5 --folds 5 --epochs 100 --device cuda:0
```

For more options:

```shell
python train.py --help
```


## ECNet-Ox Benchmark

The oxidation-state-aware benchmark uses:

```text
data/datasets/MP_ox_integer_valid.csv
utils/elec_config_oxidation_state.csv
```

It can be run by:

```shell
python train_ox.py --name ECNet_Ox_MP --path data/datasets/MP_ox_integer_valid.csv --folds 5 --epochs 100 --device cuda:0
```

The corresponding split files are provided in `data/splits-ox/`.

## Requirements

Main dependencies include PyTorch, NumPy, pandas, scikit-learn, pymatgen, matminer, and SMACT. See `requirements.txt` for details.

## Citation

If you use this code or data, please cite the ECNet manuscript.

## License

This project is released under the MIT License.
