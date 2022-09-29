# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


n = 10000
LAMBDA = 10
epsilon_form = np.random.uniform(low = 0, high = 1, size = n)

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

x_exponential = -np.log(epsilon_form) / LAMBDA

def show_KDE_plot():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.distplot(x_exponential, ax = ax, color = 'darkviolet', label = '$\lambda$ = 10')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'Частота')
    ax.set_title(u'Гістограма згенерованого експоненціального розподілу');
    ax.legend()
    plt.show()

# show_KDE_plot()
# print(f'Math_Expect = {x_exponential.mean()}')
# print(f'Standard_Deviation = {x_exponential.std(ddof = 1)}')
# print(f'Variance = {x_exponential.std(ddof = 1) ** 2}')

def generate_exponential_distribution(lambd_a, n = 10000):
    epsilon_form = np.random.uniform(low = 0, high = 1, size = n)
    return -np.log(epsilon_form) / lambd_a

# Експоненціальний інтегральний розподіл (Cumulative Distribution)
def use_cumulative_distr(x, l):
    return 1 - np.exp(-l * x)

def show_CD_plot():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    x = np.linspace(0, 1, 100000)
    sns.lineplot(x_exponential, use_cumulative_distr(x_exponential, 1 / ((x.mean() + x.std(ddof = 1)) / 2)), ax = ax, color = 'darkviolet', label = '$F(x)$')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'F(x)')
    ax.set_title(u'Застосування функції Експоненціального Інтегрального Розподілу до згенерованих даних');
    ax.legend();
    plt.show()
# show_CD_plot()


def create_exponential_bins(a, b, bins_count = 30):
    observed_frequency = {}
    expected_frequency = {}
    start = a.min()
    finish = a.max() + 1e-9
    h = (finish - start) / bins_count
    temp = start
    i = 0
    while finish >= temp:
        observed_frequency[i] = np.sum((a >= temp) & (a < (temp + h)))
        p = np.exp(-b*temp) - np.exp(-b * (h + temp))
        expected_frequency[i] = a.size * p
        i += 1
        temp += h
    return normilize_exponential_bins(observed_frequency, expected_frequency)

def normilize_exponential_bins(observed_frequency, expected_frequency):
    assert len(expected_frequency) > 2 or len(observed_frequency) > 2
    for i in sorted(observed_frequency.keys(), reverse = True)[:-1]:
        if expected_frequency[i] <= 5 or observed_frequency[i] <= 5:
            observed_frequency[i - 1] += observed_frequency[i]
            expected_frequency[i - 1] += expected_frequency[i]
            del observed_frequency[i], expected_frequency[i]  
    return observed_frequency, expected_frequency

def compliance_check():
    alpha = 0.05
    observed_frequency, expected_frequency = create_exponential_bins(x_exponential, 1 / ((x_exponential.mean() + x_exponential.std(ddof = 1)) / 2))
    # observed_frequency, expected_frequency = create_exponential_bins(x_exponential, 2) # хибне значення параметра
    observed_frequency, expected_frequency = create_exponential_bins(x_exponential, 5) # хибне значення параметра
    result, p = stats.chisquare(list(observed_frequency.values()), list(expected_frequency.values()), ddof = 1)
    
    if p < alpha:
        print('Нульова гіпотеза з параметром Alpha = 0.05 не доказана')
    else:
        print('Нульова гіпотеза про розподіл данних з заданим параметром успішно доказана')

    print('Можливість похибки (p_value): %f' % round(p, 5))
    print('Значення статистики: %f' % round(result, 5))

# compliance_check()


LAMBDA = 1.2
epsilon_form = np.random.uniform(low = 0, high = 1, size = n)

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

x_exponential = -np.log(epsilon_form) / LAMBDA
# show_CD_plot()
# compliance_check()