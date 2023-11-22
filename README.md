<p align="center">
  <img alt="RetNet Architecture" src="https://raw.githubusercontent.com/adosar/retnet/master/images/retnet.png" width="80%"/>
</p>

<h4 align="center">
  
[![Requires Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=yellow&label=Python&labelColor=black&color=blue)](https://www.python.org/downloads/)
[![Requires PyTorch 2.1.0|2.0.1+cu118](https://img.shields.io/badge/PyTorch-2.1.0|2.0.1+cu118-blue?logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Static Badge](https://img.shields.io/badge/Figshare-data-blue?logo=figshare)](https://figshare.com/articles/dataset/RetNet/24598845)

</h4>

This repository contains the necessary `Python` code to train the `RetNet` architecture[^1].

A `PyTorch` implementation of `RetNet` can be found on `model.py` module.

## 🚀 Training RetNet

The following example is used to train `RetNet` on the University of Ottawa
database[^2] for predicting CO<sub>2</sub> uptake. 

> [!IMPORTANT]
> **It is strongly recommended to run all the scripts inside a virtual environment.**

### Clone the repository

```bash
git clone https://github.com/adosar/retnet
cd retnet
```

### Dependencies

1. Create a virtual environment:
	```bash
	python -m venv <venvir_name>
	```
2. Activate it:
	```bash
	source <venvir_name>/bin/activate
	```
3. Install the dependencies:
	```bash
	(<venvir_name>) pip install -r requirements.txt
	```

### Collect the data

The following directory structure is required prior to training:
```bash
data/
├── MOFs
   ├── all_MOFs_screening_data.csv
   ├── batch_train
   │   ├── clean_names.json
   │   └── clean_voxels.npy
   └── batch_val_test
       ├── clean_names.json
       └── clean_voxels.npy
```

To achieve that:

1. Get the [inputs](https://figshare.com/articles/dataset/RetNet/24598845):
	```bash
	wget -O- 'https://figshare.com/ndownloader/files/43220463' | tar -xzvf-
	```
> [!WARNING]  
> If you use any of this data in your research work, you should cite the original work[^1].
   
2. Get the [labels](https://archive.materialscloud.org/record/2018.0016/v3):
 	```bash
  	wget -O- 'https://archive.materialscloud.org/record/file?filename=screening_data.tar.gz&record_id=62' | tar -xzvf- -C data/MOFs
  	```
> [!WARNING]  
> If you use any of this data in your research work, you should cite the original work[^2].


###  Train the model
Check the comments  in `training.py` to customize the training phase on your needs.
```bash 
(<venvir_name>) python training.py
```
* GPU training time: 40s per epoch on Nvidia GTX 1650 Super
* CPU training time: 257s per epoch on Intel i5 8400

> [!TIP]
> If you want to use a GPU but the VRAM is not enough, try decreasing the batch size to a value smaller than 64.

## 📰 Citing
If you use the `RetNet` architecture in your research work or any of the scripts of this repository, please consider citing:
> Currently N/A.

[^1]: DOI currently N/A.

[^2]: Boyd, P.G., Chidambaram, A., García-Díez, E. _et al._
Data-driven design of metal–organic frameworks for wet flue gas CO<sub>2</sub> capture.
 _Nature_ **576**, 253–256 (2019). https://doi.org/10.1038/s41586-019-1798-7
