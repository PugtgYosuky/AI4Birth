import time
import datetime
import os
import pandas as pd
import numpy as np
import json

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 500

import shap 

IMPORTANCES_DIR = 'importances'
PREDICTIONS_DIR = 'predictions'

DEFAULT_MODELS = [
  ["SVC",{}],
#   ["GaussianNB", {}],
  ["XGBClassifier",{}],
  ["AdaBoostClassifier",{}],
#   ["DecisionTreeClassifier",{}],
  ["MLPClassifier",{}],
  ["RandomForestClassifier",{}],
#   ["KNeighborsClassifier",{}],
  ["LogisticRegression",{}]
]

def create_dirs(src_path):
    """Create directories for the given path"""
    if not os.path.exists(src_path):
        os.makedirs(src_path)
    
    # create experience directory
    exp_dir = os.path.join(src_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+str(np.random.random()*np.random.random()))
    os.makedirs(exp_dir)
    predictions_dir = os.path.join(exp_dir, PREDICTIONS_DIR)
    os.makedirs(predictions_dir)
    importances_dir = os.path.join(exp_dir, IMPORTANCES_DIR)
    os.makedirs(importances_dir)
    return exp_dir


def run_experiment(train_df, test_df, config, exp_dir, results, seed, fold, shap_dict):
    """Run a single experiment"""
    # get target columns
    y_train = train_df.pop(config['target'])
    y_test = test_df.pop(config['target'])
    # reduce the number of features to the selected ones
    if config.get('features_to_use', None):
        train_df = train_df[config['features_to_use']].copy()
        test_df = test_df[config['features_to_use']].copy()
    # preprocess data
    pipeline = create_processing_pipeline(config, train_df)
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)
    # train models
    if config['models'] == 'all':
        models_to_run = DEFAULT_MODELS
    else:
        models_to_run = config['models']
    
    if config['use_shap'] and not shap_dict:
        # empty dict to store shap values
        for model_name, _ in models_to_run:
            shap_dict[model_name] = {
                'shap_values': [],
                'shap_df': []
            }

    for (model_name, model_params) in models_to_run:
        model = create_model(model_name, model_params, seed)
        print(f"Running {model_name} on fold {fold}")
        # train model
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        train_time = time.time() - start_time

        if config['use_shap']:
            # compute shap values
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
            shap_dict[model_name]['shap_values'].append(shap_values)
            shap_dict[model_name]['shap_df'].append(X_test)
            
            # save feature importance
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({'Feature':X_test.columns, 'Importance':mean_shap_values} )
            feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
            # save feature importance
            feature_importance.to_csv(os.path.join(exp_dir, IMPORTANCES_DIR, f'feature_importance_{model_name}_fold_{fold}.csv'), index=False)

            # save shap plot and data
            shap.summary_plot(shap_values, X_test, show=False, max_display=15)
            # plt.xticks(ticks=[-1, 0, 1], labels=["VD", "", "CS"]) 
            plt.xlabel('')
            plt.savefig(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_fold_{fold}_shap_values.png'), transparent=False)
            plt.savefig(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_fold_{fold}_shap_values.pdf'), transparent=False)
            plt.close()
            shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
            shap_df.to_csv(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_fold_{fold}_shap_values.csv'), index=False)
            X_test.to_csv(os.path.join(exp_dir, IMPORTANCES_DIR, f'{model_name}_fold_{fold}_shap_df.csv'), index=False)
            
        
        # save predictions
        save_predictions(y_test, y_pred, y_proba, model_name, fold, results, train_time, exp_dir)


def create_processing_pipeline(config, train_df):
    """Create processing pipeline"""
    # define features

    numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train_df.select_dtypes(include=['object', 'bool']).columns

    # create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
    ])

    # create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    return preprocessor

def create_model(model_name, model_params, seed):
    """Create a model"""
    if model_name == 'SVC':
        from sklearn.svm import SVC
        model = SVC(random_state=seed, probability=True, **model_params)
    elif model_name == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(**model_params)
    elif model_name == 'XGBClassifier':
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=seed, **model_params)
    elif model_name == 'AdaBoostClassifier':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(random_state=seed, **model_params)
    elif model_name == 'DecisionTreeClassifier':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=seed, **model_params)
    elif model_name == 'MLPClassifier':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(random_state=seed, **model_params)
    elif model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=seed, **model_params)
    elif model_name == 'KNeighborsClassifier':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**model_params)
    elif model_name == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=seed, **model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def save_predictions(y_test, y_pred, y_proba, model_name, fold, results, train_time, exp):
    """ Save predictions and metrics"""
    # save predictions in file
    columns = columns = ['y_true', 'y_pred'] + [f'y_proba_{i}' for i in range(y_proba.shape[1])]
    predictions = pd.DataFrame(np.array([y_test, y_pred] + y_proba.T.tolist()).T, columns=columns)
    predictions.to_csv(os.path.join(exp, PREDICTIONS_DIR, f'{model_name}_fold_{fold}.csv'), index=False)

    # save metrics
    model_metrics = results.get(model_name, {
        'train_time': [],
        'balanced_accuracy': [],
        'roc_auc_score' : [],
        'recall_weighted' : [],
        'f1_weighted' : [],
        'precision_weighted' : [],
        'matthews_corrcoef' : [],
        'specificity' : [],
        'NPV' : [],
        'tp' : [],
        'tn' : [],
        'fp' : [],
        'fn' : [],
    })

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

    model_metrics['train_time'].append(train_time)
    model_metrics['balanced_accuracy'].append(metrics.balanced_accuracy_score(y_test, y_pred))
    model_metrics['roc_auc_score'].append(metrics.roc_auc_score(y_test, y_proba[:,1]))
    model_metrics['recall_weighted'].append(metrics.recall_score(y_test, y_pred, average='weighted'))
    model_metrics['f1_weighted'].append(metrics.f1_score(y_test, y_pred, average='weighted'))
    model_metrics['precision_weighted'].append(metrics.precision_score(y_test, y_pred, average='weighted'))
    model_metrics['matthews_corrcoef'].append(metrics.matthews_corrcoef(y_test, y_pred))
    model_metrics['specificity'].append(tn / (tn+fp))
    model_metrics['NPV'].append(tn / (tn + fn))
    model_metrics['tp'].append(tp)
    model_metrics['tn'].append(tn)
    model_metrics['fp'].append(fp)
    model_metrics['fn'].append(fn)

    results[model_name] = model_metrics

class NpEncoder(json.JSONEncoder):
    """
    Encoder used for saving JSON files
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)
        