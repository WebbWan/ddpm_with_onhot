import numpy as np
from scipy.stats import entropy


def caculate_kl_divergence(original_data, generated_data):
    hist_original, bin_edges_original = np.histogram(original_data, bins=2, density=True)
    hist_generated, bin_edges_generated = np.histogram(generated_data, bins=2, density=True)
    kl_divergence = entropy(hist_original, hist_generated)
    return kl_divergence


data1 = np.load("./result/raw_data/penguins_size_sample/0/penguins_size_sample_raw_class0_attr.npy")
data2 = np.load("./result/synthetic_data/penguins_size_sample/0/penguins_size_sample_syn_class0_attr.npy")

print(caculate_kl_divergence(data1, data2))
