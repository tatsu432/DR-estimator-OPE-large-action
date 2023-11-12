# Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces

This repository is for the exeriment conducted in ["Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces"](https://arxiv.org/abs/2308.03443) ([IEEE SSCI 2023](https://attend.ieee.org/ssci-2023/))" by [Tatsuhiro Shimizu](https://ss1.xrea.com/tshimizu.s203.xrea.com/works/index.html) and [Laura Forastiere](https://ysph.yale.edu/profile/laura-forastiere/).

## Abstract

We study Off-Policy Evaluation (OPE) in contextual bandit settings with large action spaces. The benchmark estimators suffer from severe bias and variance tradeoffs. Parametric approaches suffer from bias due to difficulty specifying the correct model, whereas ones with importance weight suffer from variance. To overcome these limitations, [Marginalized Inverse Propensity Scoring (MIPS)](https://arxiv.org/abs/2202.06317) was proposed to mitigate the estimator's variance via embeddings of an action. To make the estimator more accurate, we propose the doubly robust estimator of MIPS called the Marginalized Doubly Robust (MDR) estimator. Theoretical analysis shows that the proposed estimator is unbiased under weaker assumptions than MIPS while maintaining variance reduction against IPS, which was the main advantage of MIPS. The empirical experiment verifies the supremacy of MDR against existing estimators.

## Citation

```
@article{shimizu2023doubly,
  title={Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces},
  author={Shimizu, Tatsuhiro and Forastiere, Laura},
  journal={arXiv preprint arXiv:2308.03443},
  year={2023}
}
```


## Requirements and Setup
```bash
# clone the repository
git clone https://github.com/tatsu432/DR-estimator-OPE-large-action.git
```

The versions of Python and necessary packages are specified as follows.

```
[tool.poetry.dependencies]
python = ">=3.9,<3.10"
obp = "0.5.5"
scikit-learn = "1.0.2"
pandas = "1.3.5"
scipy = "1.7.3"
numpy = "^1.22.4"
matplotlib = "^3.5.2"
seaborn = "^0.11.2"
hydra-core = "1.0.7"
```


### Section 5: Synthetic Data Experiment
```
# How does MDR perform with varying number of actions?
src/synthetic/main_n_actions.ipynb
```
<img width="400" alt="number of actions" src="result/new/n_actions.png.png">
<img width="400" alt="number of actions" src="result/new/n_actions_2.png.png">

```
# How does MDR perform with varying number of samples?
src/synthetic/main_n_rounds.ipynb
```
<img width="400" alt="number of samples" src="result/new/n_rounds.png.png">
<img width="400" alt="number of samples" src="result/new/n_rounds_2.png.png">

```
<!-- # How does MDR perform with varying beta?
src/synthetic/main_beta.ipynb
```
![beta](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/32c10899-2421-4c86-b9e3-6b097302fee6)

![beta_log](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/777fac76-cf91-497a-be48-91b9ae0ac798)

```
# How does MDR perform with varying epsilon?
src/synthetic/main_epsilon.ipynb
```
![epsilon](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/1f05a862-5465-44df-99c8-ad916bc0fb22)
![epsilon_log](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/7f79501d-53f2-4e15-9609-0dfb6d14d5a0)

```
# How does MDR perform with varying number of deficient actions?
src/synthetic/main_n_decifient_actions.ipynb
```
![n_deficient_actions](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/db68103f-a349-4a2d-ae73-5c15e390141d)
![n_deficient_actions_log](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/2e9605a9-9652-4a9b-be30-6cde5dd3b2e0)

```
# How does MDR perform with varying standard deviation of reward?
src/synthetic/main_reward_std.ipynb
```
![reward_std](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/b31a4f06-1905-433b-a60e-fb1ff6617a83)
![reward_std_log](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/c4e4a05a-16aa-4efe-8f23-1dda27ed7558)

```
# How does MDR perform with varying number of category dimensions?
src/synthetic/main_n_cat_dim.ipynb
```
![n_cat_dim](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/4632a76e-c9bd-46a1-833e-d3520300dd78)
![n_cat_dim_log](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/d900c406-7f29-4ab6-a952-804913e44d66)

```
# How does MDR perform with varying number of categories per dimension?
src/synthetic/main_n_cat_per_dim.ipynb

```
![n_cat_per_dim](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/2c225c4f-6749-48d4-958d-dd5a68734881)
![n_cat_per_dim_log](https://github.com/tatsu432/DR-estimator-OPE-large-action/assets/80372303/2636a1fa-bbf2-4383-b462-525602189b92) -->


