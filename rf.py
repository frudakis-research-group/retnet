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

with open('data/MOFs/batch_train/clean_names.json', 'r') as fhand:
    mof_train = json.load(fhand)['names']

with open('data/MOFs/batch_val_test/clean_names.json', 'r') as fhand:
    mof_test = json.load(fhand)['names']

column_names = [
    'MOFname', 'CO2_uptake_P0.15bar_T298K [mmol/g]',
    'heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]',
    'volume [A^3]', 'weight [u]', 'surface_area [m^2/g]',
    'void_fraction', 'void_volume [cm^3/g]', 'largest_free_sphere_diameter [A]',
    'largest_included_sphere_along_free_sphere_path_diameter [A]',
    'largest_included_sphere_diameter [A]',
    ]

df = pd.read_csv('data/MOFs/all_MOFs_screening_data.csv')
df.set_index('MOFname', inplace=True)
features = column_names[3:]
target = column_names[1]

#with open('data/COFs/batch_train/clean_names.json', 'r') as fhand:
#    mof_train = json.load(fhand)['names']
#
#with open('data/COFs/batch_val_test/clean_names.json', 'r') as fhand:
#    mof_test = json.load(fhand)['names']

#df = pd.read_csv('data/COFs/COFs_low_pressure.csv')
#df.set_index('COFname', inplace=True)
#features = ['dompore', 'maxpore', 'voidfraction', 'gsurface', 'density']
#target = 'adsV_CH4_5.8b'

reg = RandomForestRegressor(n_jobs=-1)

df_test = df.loc[mof_test]
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.dropna(inplace=True)

X_test = df_test.loc[:, features]
y_test = df_test.loc[:, target]

train_sizes = [
        100, 500, 1_000, 2_000, 5_000,
        10_000, 20_000, len(mof_train)
        ]
 
for size in train_sizes:
    df_train_tmp = df.loc[mof_train[:size]]

    df_train = df_train_tmp.replace([np.inf, -np.inf], np.nan)
    df_train.dropna(inplace=True)

    X_train = df_train.loc[:, features]
    y_train = df_train.loc[:, target]

    reg.fit(X_train, y_train)

    print(size, r2_score(y_test, reg.predict(X_test)))

    # Save the trained model.
    #with open('rf_model.pkl', 'wb') as fhand:
    #    joblib.dump(reg, fhand)
