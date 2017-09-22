import pandas as pd

from phantom_helpers import plot_errors

# methods = ['dipfit', 'lcmv', 'music']
methods = ['dipfit', 'music', 'MxNE_depth_95', 'irMxNE_depth_95']

maxfilter = 'True'
dfs = []

for m in methods:
    data = pd.read_csv('results/phantom_errors_%s.csv' % m)
    print(data)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'], axis=1)
    data = data.query('maxfilter == "%s"' % maxfilter).copy()
    data["maxfilter"] = m
    dfs.append(data)

errors = pd.concat(dfs, axis=0, ignore_index=True)

plot_errors(errors, "mne_%s" % maxfilter, postfix="",
            ylim=(0, 15), xlabel_mapping=None)
