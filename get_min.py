import pandas as pd
import numpy as np

df_test = pd.read_csv('mlp_grid_search_results.csv')
avg = df_test[["test_avg_distance"]].values
min_avg = np.min(avg)
print(min_avg)
