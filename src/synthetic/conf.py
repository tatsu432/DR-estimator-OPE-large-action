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

n_actions=2000
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

n_seeds = 20