## Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces
#### in Proceedings of the 2023 IEEE Symposium Series on Computational Intelligence
- Author: Tatsuhiro Shimizu\*1 and Laura Forastiere*2
-  Affiliation: *1 Waseda University, *2 Yale University
-   ORCID: *1 0009-0009-9746-3346
-   E-mail: *1 t.shimizu432@akane.waseda.jp
-   arXiv preprint: [https://arxiv.org/abs/2308.03669](https://arxiv.org/abs/2308.03443)
-   Abstract: We study Off-Policy Evaluation (OPE) in contextual bandit settings with large action spaces. The benchmark estimators suffer from severe bias and variance tradeoffs. Parametric approaches suffer from bias due to difficulty specifying the correct model, whereas ones with importance weight suffer from variance. To overcome these limitations, Marginalized Inverse Propensity Scoring (MIPS) was proposed to mitigate the estimator's variance via embeddings of an action. To make the estimator more accurate, we propose the doubly robust estimator of MIPS called the Marginalized Doubly Robust (MDR) estimator. Theoretical analysis shows that the proposed estimator is unbiased under weaker assumptions than MIPS while maintaining variance reduction against IPS, which was the main advantage of MIPS. The empirical experiment verifies the supremacy of MDR against existing estimators.

This is the Python code for the Marginalized Doubly Robust estimator.

- See MDR_OPE_large_action_space.ipynb for the comprehensive experiment
