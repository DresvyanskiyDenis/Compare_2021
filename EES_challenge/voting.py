import os
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import recall_score
from geneticalgorithm import geneticalgorithm as ga

def read_csv_file_with_predictions(path:str)->pd.DataFrame:
    df=pd.read_csv(path)
    return df

def calculate_weighted_voting(predictions:np.ndarray, weights:np.ndarray)->int:
    """Calculates weighted voting from proviede predictions and weights.
    predictions have shape (num_predictors,)
    weights have shape (num_predictions, num_classes). Thus, each predictor has its own weight on each class.

    :param predictions: np.ndarray
            predictions with shape (num_predictors,)
    :param weights: np.ndarray
            weights for each class of each predictor. Shape is (num_predictors, num_classes)
    :return: int
            predicted class
    """
    num_classes=weights.shape[0]
    class_scores=np.zeros(num_classes)
    for class_idx in range(num_classes):
        who_voted_for_this_class=predictions==class_idx
        class_score=(1.*weights[class_idx,who_voted_for_this_class]).sum()
        class_scores[class_idx]=class_score
    predicted_class=class_scores.argmax()
    return predicted_class


def do_weighted_prediction_to_all_instances(predictions:np.ndarray, weights:np.ndarray)-> np.ndarray:
    weighted_predictions=np.zeros((predictions.shape[0],))
    for i in range(weighted_predictions.shape[0]):
        weighted_predictions[i]=calculate_weighted_voting(predictions[i], weights)
    return weighted_predictions

def do_major_voting_to_all_instances(predictions:np.ndarray)->np.ndarray:
    predictions=mode(predictions, axis=1)[0]
    return predictions

def save_predictions(path:str, predictions:np.ndarray, filenames:np.ndarray, predictions_filename:str='predictions.csv')->None:
    if len(predictions.shape)==1:
        predictions=predictions[..., np.newaxis]
    if len(filenames.shape)==1:
        filenames=filenames[..., np.newaxis]
    concatenated_data=np.concatenate([filenames, predictions], axis=1)
    writing_df=pd.DataFrame(data=concatenated_data, columns=['filename', 'prediction'])
    writing_df.to_csv(os.path.join(path, predictions_filename), index=False)

def get_best_weights_on_predictions(predictions:np.ndarray, ground_truth_labels:np.ndarray,
                                    num_generated_weights:int=10000, num_classes:int=3,
                                    return_predictions:bool=False):
    # generate weights
    weights = np.random.dirichlet(alpha=np.ones((predictions.shape[1],)), size=(num_generated_weights, num_classes))
    best_UAR = 0
    best_weights = None
    best_predictions = None
    # find best weights
    for weights_idx in range(weights.shape[0]):
        current_predictions = do_weighted_prediction_to_all_instances(predictions, weights[weights_idx]).reshape(
            (-1,))
        metric = recall_score(ground_truth_labels, current_predictions, average='macro')
        if metric > best_UAR:
            #print('new best weights are found. UAR:%f.Weights:%s.' % (metric, weights[weights_idx]))
            best_UAR = metric
            best_weights = weights[weights_idx]
            best_predictions = current_predictions
    print('Found Best UAR:%f, Best weights:%s' % (best_UAR, best_weights))
    if return_predictions:
        return best_weights, best_predictions
    return best_weights

def adjust_weights_on_all_folds_except_one(folds:List[pd.DataFrame],num_exception:int,
                                           num_generated_weights: int = 10000, num_classes: int = 3,
                                           return_predictions: bool = False, return_weights:bool=True
                                           ):
    # concatenate ground truth and train fold predictions
    folds=folds.copy()
    valid_fold=folds.pop(num_exception)
    train_folds=folds
    train_predictions=[]
    ground_truth_labels=[]
    for train_folds in train_folds:
        filenames = train_folds.iloc[:, 0].values
        predictions_values = train_folds.iloc[:, 1:-2].values
        ground_truth_labels_fold = train_folds.iloc[:, -1].values.reshape((-1,))
        train_predictions.append(predictions_values)
        ground_truth_labels.append(ground_truth_labels_fold)
    # concatenate collected predictions and ground_truth labels
    train_predictions=np.concatenate(train_predictions, axis=0)
    ground_truth_labels=np.concatenate(ground_truth_labels, axis=0)
    # find best weights
    best_weights = get_best_weights_on_predictions(train_predictions, ground_truth_labels,
                                                   num_classes=num_classes, num_generated_weights=num_generated_weights,
                                                   return_predictions=False)
    # recalculate valid predictions according found best weights on training set
    predictions_valid=valid_fold.iloc[:, 1:-2].values
    predictions_valid=do_weighted_prediction_to_all_instances(predictions_valid, best_weights).reshape(
            (-1,))
    # calculate UAR on with found best weights
    ground_truth_valid = valid_fold.iloc[:, -1].values.reshape((-1,))
    metric=recall_score(ground_truth_valid, predictions_valid, average='macro')
    return_things=[]
    return_things.append(metric)
    if return_predictions:
        return_things.append(predictions_valid)
    if return_weights:
        return_things.append(best_weights)
    return tuple(return_things)


if __name__=='__main__':
    path_to_predictions='D:\Downloads\ensemble_folds'
    path_to_save='weighted_predictions'
    num_classes=3
    num_weights=1000000
    np.set_printoptions(precision=5)

    cv_fold_prediction_filenames=os.listdir(path_to_predictions)
    # read all folds
    folds=[]
    for cv_filename in cv_fold_prediction_filenames:
        folds.append(read_csv_file_with_predictions(os.path.join(path_to_predictions, cv_filename)))
    # iterate across each fold
    for fold_idx in range(len(folds)):
        metric, weighted_predictions, best_weights=adjust_weights_on_all_folds_except_one(folds=folds,num_exception=fold_idx,
                                           num_generated_weights=10000, num_classes = 3,
                                           return_predictions= True)
        print('Fold %i, metric:%f'%(fold_idx+1,metric))
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        save_predictions(path_to_save, weighted_predictions, folds[fold_idx].iloc[:, 0].values,
                         predictions_filename='fold_%i_predictions.csv'%(fold_idx+1))
        np.savetxt(os.path.join(path_to_save, 'fold_%i_weights.txt'%(fold_idx+1)), best_weights)