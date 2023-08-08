import numpy as np
from utils_ds.math_tool.gaussain.my_gaussPDF import my_gaussPDF
# havent been tested yet


def posterior_probs_gmm(x, gmm, type):
    N = len(x)
    M = len(x[0])

    # Unpack gmm
    Mu = gmm.Mu
    Priors = gmm.Priors
    Sigma = gmm.Sigma
    K = len(Priors)
    # Compute mixing weights for multiple dynamics
    Px_k = np.zeros((K, M))

    # Compute probabilities p(x^i|k)
    for k in np.arange(K):
        Px_k[k, :] = my_gaussPDF(x, Mu[:, k].reshape(N, 1), Sigma[k, :, :])

    # Compute posterior probabilities p(k|x) -- FAST WAY --- %%%
    alpha_Px_k = np.repeat(Priors.reshape(len(Priors),1), M, axis=1) * Px_k

    if type == 'norm':
        Pk_x = alpha_Px_k / np.repeat(np.sum(alpha_Px_k, axis=0, keepdims=True), K, axis=0)
    else:
        Pk_x = alpha_Px_k

    return Pk_x

# Note:
# 这个函数首先计算了一个K*M的矩阵，这个矩阵的列是当前点属于各个cluster的概率（基于point）然后alpha_Px_k是将prior扩张成K*M，乘上Pk_x
# 就能够得到 每个点属于每个cluster的概率,这里的norm就是 P（z = k) / (sum from i = 1 to K) P(z = i)
# 该函数最终返回一个K * M的矩阵，表示每个点属于每个类的概率

# np.sum([[0, 1], [0, 5]], axis=0)
# array([0, 6])
# np.sum([[0, 1], [0, 5]], axis=1)
# array([1, 5])
# sum 沿着什么axis就是沿着哪里加，比如0轴是列，axis=0就是沿着列相加


