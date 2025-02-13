# Description of the config

## Parameters

- *cv* [int] : Number of cross-validation k-folds. If *cv* = 1, it will need to have *train_dataset* and *test_dataset* defined to do train-test evaluation. If *cv* > 1, it requires to have *train_dataset* defined.
- *train_dataset* [str]: Path to the train dataset csv
- *test_dataset* [str]: Path to the test dataset csv
- *target* [str]: Name of the target column of the dataset
- *use_shape* [bool]: Flag to describe if SHAP should be applied or not.
- *models* [list or str]: List with the models to evaluate or *all*. If a list is provided, each item should contain a sub-list where the first element is the sklearn name of the model, and the second is a dictionary containing each parameter name and value
- *save_dir* [str]: Path to the directory where to save results
