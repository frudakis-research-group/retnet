<p align="center">
  <img alt="RetNet Architecture" src="https://raw.githubusercontent.com/adosar/retnet/master/images/forward_pass.png" width="20%"/>
</p>

<h4 align="center">
  
[![Requires Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=yellow&label=Python&labelColor=black&color=blue)](https://www.python.org/downloads/)
![Requires PyTorch 2.1.0](https://img.shields.io/badge/PyTorch-2.1.0-blue?logo=pytorch)

</h4>

This repository contains the necessary `python` scripts to train the `RetNet` architecture[^1].

A `PyTorch` implementation of `RetNet` can be found on `model.py` module.

## 🚀 Training RetNet

The following example is used to train `RetNet` on the University of Ottawa database[^2] for predicting CO~2~ uptake. 

By modifying `training.py` you can also train `RetNet` on the COFs dataset created by Mercado et al.[^3] for predicting CH~4~ uptake.

**It is strongly recommended to run all the scripts inside a virtual environent.**

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

### Clone the repository

```bash
git clone https://github.com/adosar/retnet
```

### Collect the data

The following directory structure is required prior to training:
```bash
data/
├── COFs
│   ├── batch_train
│   │   ├── clean_names.json
│   │   └── clean_voxels.npy
│   ├── batch_val_test
│   │   ├── clean_names.json
│   │   └── clean_voxels.npy
│   └── COFs_low_pressure.csv
└── MOFs
    ├── all_MOFs_screening_data.csv
    ├── batch_train
    │   ├── clean_names.json
    │   └── clean_voxels.npy
    └── batch_val_test
        ├── clean_names.json
        └── clean_voxels.npy

7 directories, 10 files
```

To achieve that:
```bash
cd retnet
curl link | tar -xvf
```

###  Train the model
Check the comments  in `training.py` to customize the training phase on your needs.[^3]
```bash 
(<venvir_name>) python training.py
```

## 📰 Cite
If you use the `RetNet` architecture in your research work or any of the scripts of this repository, please consider citing:
> Currently N/A.

[^1]: DOI currently N/A.

[^2]: Boyd, P.G., Chidambaram, A., García-Díez, E. _et al._
Data-driven design of metal–organic frameworks for wet flue gas CO~2~ capture.
 _Nature_ **576**, 253–256 (2019). https://doi.org/10.1038/s41586-019-1798-7

[^3]: In Silico Design of 2D and 3D Covalent Organic Frameworks for Methane Storage Applications.
Rocío Mercado, Rueih-Sheng Fu, Aliaksandr V. Yakutovich, Leopold Talirz, Maciej Haranczyk, and Berend Smit.
Chemistry of Materials **2018** _30_ (15), 5069-5086. https://doi.org/10.1021/acs.chemmater.8b01425
