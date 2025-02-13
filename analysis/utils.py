import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from IPython.display import display

def get_mean_results(experiment_path):
    src_data = get_raw_results(experiment_path)
    results_models = list(src_data.keys())
    results_metrics = list(src_data[results_models[0]].keys())
    # calculate mean and std
    data = src_data.copy()
    for model in results_models:
        for metric in results_metrics:
            data[model][metric] = np.mean(src_data[model][metric])
            data[model][metric + '_std'] = np.std(src_data[model][metric])
    df = pd.DataFrame(data).T
    return df

def get_raw_results(experiment_path):
    results = {}
    for run_number, run in enumerate(os.listdir(experiment_path)):
        try:
            with open(os.path.join(experiment_path, run, 'results.json')) as f:
                results[run_number] = json.load(f)
        except:
            pass
    results_models = list(results[run_number].keys())
    results_metrics = list(results[run_number][results_models[0]].keys())
    # initialize data structure
    src_data = {}
    for model in results_models:
        aux = {}
        for metric in results_metrics:
            aux[metric] = []
        src_data[model] = aux
    # extract data
    for run in results:
        for model in results_models:
            for metric in results_metrics:
                src_data[model][metric] += results[run][model][metric]
    return src_data

def test_normal_ks(data):
    """Kolgomorov-Smirnov"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return stats.kstest(norm_data,'norm')

def levene(group_a, group_b):
    """Test of equal variance."""
    return stats.levene(group_a, group_b)

def statistical_test(group_a, group_b, significance_level=0.05):
    # remove where data is null
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    print(f'GROUP A: {round(group_a.mean(), 1)} (±{round(group_a.std(), 1)})')
    print(f'GROUP B: {round(group_b.mean(),1)} (±{round(group_b.std(), 1)})')

    # print('- Two categories\n- different initial conditions\n\n')

    # normality tests
    norm_vaginal = test_normal_ks(group_a)
    norm_csa = test_normal_ks(group_b)
    # varience test
    variance = levene(group_a, group_b)

    # print('Normality H0: The sample follows a normal distribution')
    # print('Normality Ha: The sample does not follow a normal distribution')

    # print('Variance H0: The variance is equal ')
    # print('Variance Ha: The variance is different')
    
    # print('Normality test: grupo vaginal\n', norm_vaginal)
    # print('\nNormality test: grupo csa\n', norm_csa)
    # print('\nLevene test: grupo vaginal\n', variance)

    parametric = norm_vaginal.pvalue >= significance_level and \
        norm_csa.pvalue >= significance_level and \
        variance.pvalue >= significance_level
    
    # print('Parametric test: ', parametric)


    # choose the statistical test
    
    if parametric:
        # independent t-test
        stat_test = stats.ttest_ind(group_a, group_b, eq_var=True)
    else:
        # mann- Whitney
        stat_test = stats.mannwhitneyu(group_a, group_b)
    print('Statistical test:\n',stat_test)
    if stat_test.pvalue < 0.001:
        print('P-Value: < 0.001')
    else:
        print('P-Value:', round(stat_test.pvalue))


def categorical_analysis(series, target):
    # remove nans from the counts
    target = target.copy()
    series = series.copy()
    target = target.loc[series.notnull()]
    series = series.loc[series.notnull()]
    class_distribution = target.value_counts()
    aux = []
    for value in series.unique():    
        val = target.loc[series == value].value_counts()
        val.name = value
        aux.append(val)
    pivot = pd.DataFrame(aux)
    display(pivot)
    class_distribution = target.value_counts()
    percentages = pivot / class_distribution
    final_df = pivot.copy()
    for col in pivot.columns:
        final_df[col] = final_df[col].astype(str)
        for index in pivot.index:
            final_df.loc[index, col] = f'{pivot.loc[index, col]} ({(percentages.loc[index][col]*100).round(1)}%)'
    display(final_df)
    stat_test = stats.chi2_contingency(pivot)
    print(stat_test)
    if stat_test.pvalue < 0.001:
        print('P-Value: <0.001')
    else:
        print('P-Value:', round(stat_test.pvalue, 3))


def models_statistical_test(group_a, group_b, significance_level=0.05):
    # remove where data is null
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    # print(f'GROUP A: {round(group_a.mean(), 3)} (±{round(group_a.std(), 3)})')
    # print(f'GROUP B: {round(group_b.mean(),3)} (±{round(group_b.std(), 3)})')

    # print('- Two categories\n- different initial conditions\n\n')

    # normality tests
    norm_vaginal = test_normal_ks(group_a)
    norm_csa = test_normal_ks(group_b)
    # varience test
    variance = levene(group_a, group_b)

    # print('Normality H0: The sample follows a normal distribution')
    # print('Normality Ha: The sample does not follow a normal distribution')

    # print('Variance H0: The variance is equal ')
    # print('Variance Ha: The variance is different')
    
    # print('Normality test: grupo vaginal\n', norm_vaginal)
    # print('\nNormality test: grupo csa\n', norm_csa)
    # print('\nLevene test: grupo vaginal\n', variance)

    parametric = norm_vaginal.pvalue >= significance_level and \
        norm_csa.pvalue >= significance_level and \
        variance.pvalue >= significance_level
    
    # print('Parametric test: ', parametric)


    # choose the statistical test
    
    if parametric:
        # independent t-test
        stat_test = stats.ttest_rel(group_a, group_b, eq_var=True)
    else:
        # mann- Whitney
        stat_test = stats.wilcoxon(group_a, group_b)
    # print('Statistical test:\n',stat_test)
    pvalue = str(round(stat_test.pvalue, 3))
    if stat_test.pvalue < 0.001:
        # print('P-Value: < 0.001')
        pvalue = '< 0.001'
    # else:
        # print('P-Value:', round(stat_test.pvalue))
    return pvalue

