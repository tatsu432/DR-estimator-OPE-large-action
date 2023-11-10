import numpy as np
from obp.dataset import linear_reward_function
from obp.dataset import (SyntheticBanditDatasetWithActionEmbeds, 
                        linear_reward_function)
from obp.ope import RegressionModel
from ope import run_ope
import pandas as pd
from pandas import DataFrame
from policy import gen_eps_greedy
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

n_actions=1000
dim_context=10
beta=-1
reward_type="continuous"
n_cat_per_dim=10
latent_param_mat_dim=5
n_cat_dim=3
n_unobserved_cat_dim=0
n_deficient_actions=0
reward_function=linear_reward_function
reward_std=2.5
random_state=12345 

eps=0.05



markersize = 12





# # fastest
# n_seeds = 4
# n_test = 100
# n_train = 100
# n_actions_list = [5, 10, 20, 40, 80]
# n_train_list = [20, 40, 80]
# beta_list = [-3, 0, 3]
# epsilon_list = [0, 0.5, 1.0]
# n_cat_dim_list = [5, 10, 15]
# n_cat_per_dim_list = [5, 10, 15]
# n_deficient_actions_list = [0, 5, 10]
# reward_std_list = [1, 2, 3]

# faster
n_seeds = 20
n_test = 200000
n_train = 10000
n_actions_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
n_train_list = [500, 1000, 2000, 4000, 8000]
beta_list = [-3, -2, -1, 0, 1, 2, 3]
epsilon_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_cat_dim_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_cat_per_dim_list = [5, 10, 15, 20]
n_deficient_actions_list = [0, 100, 300, 500, 700, 900]
reward_std_list = [1, 2, 3, 4, 5, 6]

# # slower
# n_seeds = 100
# n_test = 10000
# n_train = 10000
# n_actions_list = [500, 1000, 2000, 4000, 8000]
# n_train_list = [500, 1000, 2000, 4000, 8000]
# beta_list = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
# epsilon_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# n_cat_dim_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# n_cat_per_dim_list = [5, 10, 15, 20, 25, 30]
# n_deficient_actions_list = [0, 100, 300, 500, 700, 900]
# reward_std_list = [1, 2, 3, 4, 5, 6]

# # slowest
# n_seeds = 100
# n_test = 1000
# n_train = 5000
# n_actions_list = [500, 1000, 2000, 4000, 8000]
# n_train_list = [500, 1000, 2000, 4000, 8000]
# beta_list = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
# epsilon_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# n_cat_dim_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# n_cat_per_dim_list = [5, 10, 15, 20, 25, 30, 35, 40]
# n_deficient_actions_list = [0, 100, 300, 500, 700, 900]
# reward_std_list = [1, 2, 3, 4, 5, 6]

