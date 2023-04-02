import numpy as np

np.random.seed(10)
activations = list(np.random.randn(5000) * 2)

#
print("真实期望:", np.mean(activations))
print("真实方差:", np.var(activations))


#

def variance(batch_size=20, momentum=0.3):
    batches = len(activations) // batch_size

    biased_var = []

    moving_var = []
    unbiased_var = []
    tmp = []
    m_var = 0
    s_idx = 0
    for i in range(batches):
        e_idx = i * batch_size + batch_size
        mini_batch = activations[s_idx:e_idx]

        bi_var = np.var(mini_batch)
        biased_var.append(bi_var)

        un_var = np.var(mini_batch) * len(mini_batch) / (len(mini_batch) - 1)
        tmp.append(un_var)
        unbiased_var.append(np.mean(tmp))

        m_var = momentum * m_var + (1 - momentum) * bi_var
        moving_var.append(m_var)
        s_idx = e_idx
    return unbiased_var, biased_var, moving_var


if __name__ == '__main__':
    unbiased_var, biased_var, moving_var = variance(batch_size=50, momentum=0.9)
    import matplotlib.pyplot as plt

    plt.hlines(np.var(activations), -1, len(unbiased_var) + 1, linestyles='--', label='true var')
    # plt.plot(range(len(unbiased_var)), unbiased_var, label='unbiased var')
    plt.plot(range(len(unbiased_var)), biased_var, label='mini-batch var in training')
    plt.plot(range(len(unbiased_var)), moving_var, label='moving average with momentum = 0.9')
    plt.legend()
    plt.show()
