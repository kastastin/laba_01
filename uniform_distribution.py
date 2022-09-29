# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


n = 10000
a = 5 ** 13
c = 2 ** 31
z_0 = 9
z = z_0
x_form = []

for i in range(n):
    x = z / c
    x_form.append(x)
    z = (a * z) % c
x_form = np.array(x_form)

# Гістрограма розподілу
def show_distribution_histogram():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.distplot(x_form, ax = ax, color = 'darkviolet')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'Частота')
    ax.set_title(u'Рівномірний розподіл')
    plt.show()

# show_distribution_histogram()
# print(f'Math_Expect = {x_form.mean()}\nVariance = {x_form.std(ddof = 1) ** 2}')

b_param = x_form.mean() + np.sqrt(3) * x_form.std(ddof = 1)
a_param = 2 * x_form.mean() - b_param
# print(f'Parametr_a = {round(a_param, 5)}\nParametr_b = {round(b_param, 5)}')

def generate_uniform_distribution(a, c, n = 10000):
    x_form = []
    for i in range(n):
        x = z / c
        x_form.append(x)
        z = (a * z) % c
    return np.array(x_form)

# Функція рівномірного закону розподілу (Probability Density)
def use_probability_dens(a, b, x):
    return (1 / (b - a)) * ((x >= a) & (x <= b))

def show_PD_plot():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.lineplot(x_form, use_probability_dens(a_param, b_param, x_form), ax = ax, color = 'darkviolet', label = '$f(x)$')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'f(x)')
    ax.set_title(u'Застосування рівномірного закону розподілу до згенерованих даних');
    ax.legend();
    plt.show()
# show_PD_plot() 


def create_uniform_bins(x, a, b, bins_count = 30):
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
        p = np.abs(stats.uniform(a, b).cdf(temp) - stats.uniform(a, b).cdf(h + temp))
        expected_frequency[i] = n * p
        i += 1
        temp += h
    return normilize_bins_uniform(observed_frequency, expected_frequency)

def normilize_bins_uniform(observed_frequency, expected_frequency):
    assert len(observed_frequency) > 2 or len(expected_frequency) > 2
    for i in sorted(observed_frequency.keys(), reverse=True)[:-1]:
        if observed_frequency[i] <= 5 or expected_frequency[i] <= 5:
            observed_frequency[i - 1] += observed_frequency[i]
            expected_frequency[i - 1] += expected_frequency[i]
            del observed_frequency[i], expected_frequency[i]
    
    for i in sorted(observed_frequency.keys())[:-1]:
        if observed_frequency[i] <= 5 or expected_frequency[i] <= 5:
            j = 1
            while not i + j in observed_frequency:
                j += 1
            observed_frequency[i + j] += observed_frequency[i]
            expected_frequency[i + j] += expected_frequency[i]
            del observed_frequency[i], expected_frequency[i]
    return observed_frequency, expected_frequency

def compliance_check():
    alpha = 0.05
    b = (x_form.mean() + np.sqrt(3) * x_form.std(ddof = 1))
    a = 2 * x_form.mean() - b
    observed_frequency, expected_frequency = create_uniform_bins(x_form, a, b)
    # observed_frequency, expected_frequency = create_uniform_bins(x_form, 0, 0.5) # хибне значення параметру
    # observed_frequency, expected_frequency = create_uniform_bins(x_form, 0, 0.9) # # хибне значення параметру
    result, p = stats.chisquare(list(observed_frequency.values()), list(expected_frequency.values()), ddof = 2)

    if p < alpha:
        print('Нульова гіпотеза з параметром Alpha = %d не доказана' % alpha)
    else:
        print('Нульова гіпотеза про розподіл данних з заданим параметром успішно доказана')

    print('Можливість похибки (p_value): %f' % round(p, 5))
    print('Значення статистики: %f' % round(result, 5))

# compliance_check()


a = 5 ** 19
c = 2 ** 63
z_0 = 17
z = z_0
x_form = []

for i in range(n):
    x = z / c
    x_form.append(x)
    z = (a * z) % c
x_form = np.array(x_form)

# Гістрограма розподілу
def show_distribution_histogram():
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.distplot(x_form, ax = ax, color = 'darkviolet')
    ax.set_xlabel(u'Згенеровані дані')
    ax.set_ylabel(u'Частота')
    ax.set_title(u'Рівномірний розподіл')
    plt.show()

# show_distribution_histogram()
# print(f'Math_Expect = {x_form.mean()}\nVariance = {x_form.std(ddof = 1) ** 2}')

b_param = x_form.mean() + np.sqrt(3) * x_form.std(ddof = 1)
a_param = 2 * x_form.mean() - b_param
# print(f'Parametr_a = {round(a_param, 5)}\nParametr_b = {round(b_param, 5)}')

# show_PD_plot()
# compliance_check()