## Requirements and installation
* `python >= 3.6`
* `pytorch >= 1.2.0`
	* conda install pytorch torchvision -c pytorch
* `tqdm`
	* conda install tqdm
* `tensorboardX`
	* pip install tensorboardX
* `GPy`
	* pip install GPy
* `emukit`
	* pip install emukit
* `easydict`
	* pip install easydict
* `PyYAML`
	* conda install pyyaml


## Test Data
The regression benchmark datasets are stored in data/regression_datasets.
The Bayesian optimization functions are implemented in utils/bo_function.py.
The Bayesian quadrature functions are implemented in the Emukit package.

## Generate synthetic data
python get_data_gp.py

## Train model using synthetic data
python run_exp.py -c config/train.yaml

## Run regression experiment using pretrained model
python run_exp.py -c config/regression.yaml -t

## Run Bayesian optimization experiment using pretrained model
python run_exp_bo.py -c config/bo.yaml

## Run Bayesian quadrature experiment using pretrained model
python run_exp_bq.py -c config/bq.yaml