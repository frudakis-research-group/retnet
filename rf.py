import json
import joblib
import numpy as np
import pandas as pd
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

# Load the data set.
df = pd.read_csv('data/MOFs/all_MOFs_screening_data.csv', index_col='MOFname')

# Instantiate the regressor.
reg = RandomForestRegressor(n_jobs=-1)

# Create the test set.
df_test = df.loc[mof_test]
X_test = df_test.loc[:, features]
y_test = df_test.loc[:, target]

train_sizes = [
        100, 500, 1_000, 2_000, 5_000,
        10_000, 15_000, 20_000, len(mof_train)
        ]
 
# Iterate over different training set sizes and estimate performance.
for size in train_sizes:
    df_train = df.loc[mof_train[:size]]

    X_train = df_train.loc[:, features]
    y_train = df_train.loc[:, target]

    reg.fit(X_train, y_train)

    print(size, reg.score(X_test, y_test))

    # Save the trained model.
    #with open('rf_model.pkl', 'wb') as fhand:
    #    joblib.dump(reg, fhand)
