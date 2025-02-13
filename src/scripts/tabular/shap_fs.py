""" Script to calculate the feature importance of a dataset"""

import sys
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# setup config
from sklearn import set_config
set_config(transform_output='pandas')

import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 500

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from utils import *

def run_exp(config, seed=42):
    # fix random seed
    np.random.seed(seed)
    # save config
    config['seed'] = seed

    # create directories
    exp_dir = create_dirs(config.get('save_dir', 'logs'))

    with open(os.path.join(exp_dir, 'config.json'), 'w') as file:
        json.dump(config, file, indent=4)

    results = {}
    shap_values = {}
    if config.get('cv', 5) > 1:
        # cross-validation
        df = pd.read_csv(config['train_dataset'])
        kfold = StratifiedKFold(n_splits=config['cv'], shuffle=True, random_state=seed)
        for fold, (train_indexes, test_indexes) in enumerate(kfold.split(df, df[config['target']])):
            train_df = df.iloc[train_indexes]
            test_df = df.iloc[test_indexes]
            run_experiment(
                train_df = train_df,
                test_df = test_df,
                config = config,
                exp_dir = exp_dir,
                results = results,
                seed = seed,
                fold = fold,
                shap_dict = shap_values
            )
    else:
        # train test
        train_df = pd.read_csv(config['train_dataset'])
        test_df = pd.read_csv(config['test_dataset'])
        run_experiment(
            train_df = train_df,
            test_df = test_df,
            config = config,
            exp_dir = exp_dir,
            results = results,
            seed = seed,
            fold = 1,
            shap_dict = shap_values
            )
        
    # save shap plots
    for model_name, shap_dict in shap_values.items():
        combined_shap_values = np.vstack(shap_dict['shap_values'])
        combined_df = pd.concat(shap_dict['shap_df'])
        shap.summary_plot(combined_shap_values, combined_df, show=False, max_display=15)
        # plt.xticks(ticks=[-1, 0, 1], labels=["VD", "", "CS"]) 
        plt.xlabel('')
        plt.savefig(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_shap_values_plot_folds_avg.png'), transparent=False)
        plt.savefig(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_shap_values_plot_folds_avg.pdf'), transparent=False)
        plt.close()
        # save shap values
        shap_df = pd.DataFrame(combined_shap_values, columns=combined_df.columns)
        shap_df.to_csv(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_shap_values_folds_avg.csv'), index=False)
        combined_df.to_csv(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_shap_df_folds_avg.csv'), index=False)
        # save feature importance
        mean_shap_values = np.abs(combined_shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({'Feature':combined_df.columns, 'Importance':mean_shap_values})
        feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importance.to_csv(os.path.join(exp_dir, IMPORTANCES_DIR, f'feature_importance_{model_name}_folds_avg.csv'), index=False)
    
    # save results
    with open(os.path.join(exp_dir, 'results.json'), 'w') as file:
        json.dump(results, file, indent=4, cls=NpEncoder)

    mean_results = results.copy()
    for key, value in mean_results.items():
        for sub_key, sub_value in value.items():
            mean_results[key][sub_key] = np.mean(sub_value)

    results_df = pd.DataFrame(mean_results).T
    results_df.reset_index(inplace=True, names='model')
    results_df.to_csv(os.path.join(exp_dir, 'results.csv'), index=False)

    return shap_dict


if __name__ == '__main__':

    # get config file from terminal
    with open(sys.argv[1], 'r') as file:
        config = json.load(file)

    save_path = config.get('save_dir', 'logs')
    if os.path.exists(save_path):
        print(f"Directory {save_path} already exists. Please remove it first.")
        sys.exit(1)

    seeds = [123, 987, 456, 789, 321, 654, 876, 234, 567, 890, 432, 765, 109, 876, 543, 210, 987, 345, 678, 901, 1234, 5678, 9012, 3456, 7890, 2345, 6789, 1263, 4567, 8901]
    # run experiment

    shap_values = []
    shap_dfs = []

    for exp_number, seed in enumerate(seeds):
        print(f"Running experiment with seed: {seed} [{exp_number+1}/{len(seeds)}]")
        shap_dict = run_exp(config, seed)
        shap_values += shap_dict['shap_values']
        shap_dfs += shap_dict['shap_df']
        if config.get('train_dataset') == config.get('test_dataset') and config['models'][0][0] == 'LogisticRegression':
            break

    # merge shap values of 30 seeds x 10 folds
    combined_shap_values = np.vstack(shap_values)
    combined_df = pd.concat(shap_dfs)
    shap.summary_plot(combined_shap_values, combined_df, show=False, max_display=15)
    plt.xlabel('')
    plt.savefig(os.path.join(save_path, f'Logistic_regression_avg_all_seeds_shap_values.png'), transparent=False)
    plt.savefig(os.path.join(save_path, f'Logistic_regression_avg_all_seeds_shap_values.pdf'), transparent=False)
    plt.close()
    # save shap values
    shap_df = pd.DataFrame(combined_shap_values, columns=combined_df.columns)
    shap_df.to_csv(os.path.join(save_path, f'Logistic_regression_avg_all_seeds_shap_values.csv'), index=False)
    combined_df.to_csv(os.path.join(save_path, f'Logistic_regression_avg_all_seeds_shap_df.csv'), index=False)
    
    # save feature importance after 30 runs
    mean_shap_values = np.abs(combined_shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({'Feature':combined_df.columns, 'Importance':mean_shap_values})
    feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importance.to_csv(os.path.join(save_path, 'feature_importance_avg_all_seeds.csv'), index=False)
