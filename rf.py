import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as dae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# For reproducible results.
np.random.seed(1)

# Load the names of materials used as train data.
with open('data/MOFs/batch_train/clean_names.json', 'r') as fhand:
    mof_train = json.load(fhand)['names']

# Load the names of materials used as test data.
with open('data/MOFs/batch_val_test/clean_names.json', 'r') as fhand:
    mof_test = json.load(fhand)['names'][5000:]

# Define the features and the target.
features = [
    'volume [A^3]', 'weight [u]', 'surface_area [m^2/g]',
    'void_fraction', 'void_volume [cm^3/g]', 'largest_free_sphere_diameter [A]',
    'largest_included_sphere_along_free_sphere_path_diameter [A]',
    'largest_included_sphere_diameter [A]',
    ]

target = 'CO2_uptake_P0.15bar_T298K [mmol/g]'

# Load features and labels.
df = pd.read_csv('data/MOFs/all_MOFs_screening_data.csv', index_col='MOFname')

reg = RandomForestRegressor(n_jobs=-1)

# Drop missing values.
df_test = df.loc[mof_test]
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.dropna(inplace=True)

X_test = df_test.loc[:, features]
y_test = df_test.loc[:, target]

train_sizes = [
        100, 500, 1_000, 2_000, 5_000,
        10_000, 15_000, 20_000, len(mof_train)
        ]
 
for size in train_sizes:
    df_train_tmp = df.loc[mof_train[:size]]

    df_train = df_train_tmp.replace([np.inf, -np.inf], np.nan)
    df_train.dropna(inplace=True)

    X_train = df_train.loc[:, features]
    y_train = df_train.loc[:, target]

    reg.fit(X_train, y_train)

    print(size, r2_score(y_true=y_test, y_pred=reg.predict(X_test)))

    # Save the trained model.
    #with open('rf_model.pkl', 'wb') as fhand:
    #    joblib.dump(reg, fhand)
