""" Script to analyse the number of features required for a certain performance """

import sys
import json

import pandas as pd
import numpy as np
import time
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

def run_exp(config, features_to_use, seed=42):
    # fix random seed
    np.random.seed(seed)

    # create directories
    # save config
    config['seed'] = seed
    config['features_used'] = features_to_use
    config['n_features'] = len(features_to_use)

    exp_dir = create_dirs(os.path.join(config.get('save_dir', 'logs'), f'features_{len(feature_to_use)}'))

    with open(os.path.join(exp_dir, 'config.json'), 'w') as file:
        json.dump(config, file, indent=4)

    train_features = features_to_use + [config['target']]

    results = {}
    shap_values = {}
    if config.get('cv', 5) > 1:
        # cross-validation
        df = pd.read_csv(config['train_dataset'])
        df = df[train_features].copy()
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
        train_df = train_df[train_features].copy()
        test_df = pd.read_csv(config['test_dataset'])
        test_df = test_df[train_features].copy()
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
        
    if config['use_shap']:
        # save shap plots
        for model_name, shap_dict in shap_values.items():
            combined_shap_values = np.vstack(shap_dict['shap_values'])
            combined_df = pd.concat(shap_dict['shap_df'])
            shap.summary_plot(combined_shap_values, combined_df, show=False, max_display=15)
            # plt.xticks(ticks=[-1, 0, 1], labels=["VD", "", "CS"]) 
            plt.xlabel('')
            plt.savefig(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_shap_values.png'), transparent=False)
        
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

if __name__ == '__main__':

    # get config file from terminal
    with open(sys.argv[1], 'r') as file:
        config = json.load(file)

    feature_importance = pd.read_csv(config['feature_importance'])
    feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
    for i in range(1, len(feature_importance)+1):
        # get features to use
        feature_to_use = feature_importance.Feature.iloc[:i].to_list()
        print(f"Running experiment with {len(feature_to_use)} features")
        seeds = [123, 987, 456, 789, 321, 654, 876, 234, 567, 890, 432, 765, 109, 876, 543, 210, 987, 345, 678, 901, 1234, 5678, 9012, 3456, 7890, 2345, 6789, 1263, 4567, 8901]
        # run experiment
        for exp_number, seed in enumerate(seeds):
            time.sleep(1)
            print(f"Running experiment with seed: {seed} [{exp_number+1}/{len(seeds)}]")
            run_exp(config, feature_to_use, seed)
        