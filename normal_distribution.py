# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


n = 10000
MU = 0
SIGMA = 1
epsilon_form = np.random.uniform(low = 0, high = 1, size = (n, 12))

# Гістрограма розподілу
def show_distribution_histogram():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.distplot(epsilon_form, ax = ax, color = 'darkviolet')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'Частота')
    ax.set_title(u'Рівномірний розподіл')
    plt.show()

# show_distribution_histogram()
# print(f'Math_Expect = {epsilon_form.mean()}\nVariance = {epsilon_form.std(ddof = 1) ** 2}')

x_normal = SIGMA * (epsilon_form.sum(axis = 1) - 6) + MU

def show_KDE_plot():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.distplot(x_normal, ax = ax, color = 'darkviolet', label='$\mu = 0$, $\sigma = 1$')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'Частота')
    ax.set_title(u'Гістограма згенерованого нормального розподілу');
    ax.legend()
    plt.show()
# show_KDE_plot()

# print(f'Math_Expect = {x_normal.mean()}')
# print(f'Standard_Deviation = {x_normal.std(ddof = 1)}')
# print(f'Variance = {x_normal.std(ddof = 1) ** 2}')

def generate_normal_distribution(mu, sigma, n = 10000):
    epsilon_form = np.random.uniform(low = 0, high = 1, size = (n, 12))
    return SIGMA * (epsilon_form.sum(axis = 1) - 6) + MU

# Функція рівномірного закону розподілу (Probability Density)
def use_probability_dens(x, mu, sigma):
    return np.exp(-np.power( (x - mu) / sigma, 2) / 2 ) * (1 / ( np.sqrt(np.pi * 2) * sigma ))

def show_PD_plot():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.lineplot(x_normal, use_probability_dens(x_normal, x_normal.mean(), x_normal.std(ddof = 1)), ax = ax, color = 'darkviolet', label = '$f(x)$')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'f(x)')
    ax.set_title(u'Застосування рівномірного закону розподілу до згенерованих даних');
    ax.legend();
    plt.show()
# show_PD_plot()

# Функція інтегрального розподілу (Cumulative Distribution)
def show_CD_plot():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.lineplot(x_normal, stats.norm.cdf(x_normal, x_normal.mean(), x_normal.std(ddof = 1)), ax = ax, color = 'darkviolet', label = '$F(x)$')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'F(x)')
    ax.set_title(u'Застосування функції інтегрального розподілу до згенерованих даних');
    ax.legend();
    plt.show()
# show_CD_plot()


def normilize_normally_bins(observed_frequency, expected_frequency):
    assert len(observed_frequency) > 2 or len(expected_frequency) > 2
    for i in sorted(observed_frequency.keys(), reverse = True)[:-1]:
        if observed_frequency[i] <= 5 or expected_frequency[i] <= 5:
            observed_frequency[i-1] += observed_frequency[i]
            expected_frequency[i-1] += expected_frequency[i]
            del observed_frequency[i], expected_frequency[i]
    
    for i in sorted(observed_frequency.keys())[:-1]:
        if observed_frequency[i] <= 5 or expected_frequency[i] <= 5:
            j = 1
            while not i+j in observed_frequency:
                j += 1
            observed_frequency[i+j] += observed_frequency[i]
            expected_frequency[i+j] += expected_frequency[i]
            del observed_frequency[i], expected_frequency[i]
    return observed_frequency, expected_frequency

def create_normally_bins(mu, sigma, x, bins_count = 30):
    observed_frequency = {}
    expected_frequency = {}
    start = x.min()
    finish = x.max() + 1e-9
    n = x.size
    h = (finish - start) / bins_count
    temp = start
    i = 0
    while temp <= finish:
        observed_frequency[i] = np.sum((x >= temp) & (x < (h + temp)))
        p = np.abs(stats.norm(mu, sigma).cdf(temp) - stats.norm(mu, sigma).cdf(h + temp))
        expected_frequency[i] = n * p
        i += 1
        temp += h
    return normilize_normally_bins(observed_frequency, expected_frequency)

def compliance_check():
    alpha = 0.05
    observed_frequency, expected_frequency = create_normally_bins(x_normal.mean(), x_normal.std(ddof = 1), x_normal)
    # observed_frequency, expected_frequency = create_normally_bins(0, 15, x_normal) # хибне значення параметра
    # observed_frequency, expected_frequency = create_normally_bins(10, 25, x_normal) # хибне значення параметра
    result, p = stats.chisquare(list(observed_frequency.values()), list(expected_frequency.values()), ddof = 2)

    if p < alpha:
        print('Нульова гіпотеза з параметром Alpha = %d не доказана' % alpha)
    else:
        print('Нульова гіпотеза про розподіл данних з заданим параметром успішно доказана')

    print('Можливість похибки (p_value): %f' % round(p, 5))
    print('Значення статистики: %f' % round(result, 5))

# compliance_check()


MU = 12
SIGMA = 24
epsilon_form = np.random.uniform(low = 0, high = 1, size = (n, 12))

# Гістрограма розподілу
def show_distribution_histogram():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.distplot(epsilon_form, ax = ax, color = 'darkviolet')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'Частота')
    ax.set_title(u'Рівномірний розподіл')
    plt.show()

# show_distribution_histogram()
# print(f'Math_Expect = {epsilon_form.mean()}\nVariance = {epsilon_form.std(ddof = 1) ** 2}')

x_normal = SIGMA * (epsilon_form.sum(axis = 1) - 6) + MU
# show_PD_plot()
# show_CD_plot()
compliance_check()