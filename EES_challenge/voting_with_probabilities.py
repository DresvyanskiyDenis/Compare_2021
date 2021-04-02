import os
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import recall_score

def read_predictions_oxana_format(path:str):
    pred=pd.read_csv(path)
    pred.sort_values(by=['filename'], inplace=True)
    pred=pred.iloc[1:,2:].values
    return pred

def do_weighted_prediction_to_all_instances_with_probs(predictions:List[np.ndarray], weights:np.ndarray)-> np.ndarray:
    final_predictions =predictions[0]*weights[:,0]
    for weight_idx in range(weights.shape[1]):
        final_predictions = final_predictions + predictions[weight_idx]*weights[:,weight_idx]
    final_predictions=np.argmax(final_predictions, axis=-1).reshape((-1,))
    return final_predictions


def find_best_weights(predictions:List[np.ndarray], ground_truth_labels:np.ndarray,num_weights:int=5000,):
    weights=np.random.dirichlet(alpha=np.ones((len(predictions),)), size=(num_weights, num_classes))
    best_UAR = 0
    best_weights = None
    best_predictions = None
    ground_truth_labels=ground_truth_labels.reshape((-1,))
    # find best weights
    for weights_idx in range(weights.shape[0]):
        current_predictions = do_weighted_prediction_to_all_instances_with_probs(predictions, weights[weights_idx])
        metric = recall_score(ground_truth_labels, current_predictions, average='macro')
        if metric > best_UAR:
            # print('new best weights are found. UAR:%f.Weights:%s.' % (metric, weights[weights_idx]))
            best_UAR = metric
            best_weights = weights[weights_idx]
            best_predictions = current_predictions
    print('Found Best UAR:%f, Best weights:%s' % (best_UAR, best_weights))
    return best_weights

def read_ground_truth_labels_folds(path_to_folds:str):
    folds=[]
    filenames=os.listdir(path_to_folds)
    for filename in filenames:
        full_path=os.path.join(path_to_folds, filename)
        df=pd.read_csv(full_path)
        df.sort_values(by=['filename'], inplace=True)
        folds.append(df['label'].values.reshape((-1,)))
    return folds


if __name__=='__main__':
    path_to_predictions='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2021\\Predictions\\Validation'
    path_to_ground_truth_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2021\\Predictions\\Validation\\OpenSmile_50'
    path_to_save_weights='weights'
    num_classes=3
    num_weights=5000
    np.set_printoptions(precision=5)
    # load ground truth labels:
    ground_truth_labels=read_ground_truth_labels_folds(path_to_ground_truth_labels)
    models=[]
    prediction_folders=os.listdir(path_to_predictions)
    for prediction_folder in prediction_folders:
        folds=[]
        prediction_fold_filenames=os.listdir(os.path.join(path_to_predictions, prediction_folder))
        for prediction_fold_filename in prediction_fold_filenames:
            predictions=read_predictions_oxana_format(path=os.path.join(path_to_predictions, prediction_folder,prediction_fold_filename))
            folds.append(predictions)
        models.append(folds)
    # now we have list such as:
    # [ model_1:[fold_1_predictions, fold_2_predictions, ...]
    #   model_2: [fold_1_predictions, fold_2_predictions, ...]
    #   ...
    # ]
    # let's generate weights for each fold
    num_folds=len(models[0])
    for fold_idx in range(num_folds):
        predictions_from_different_models=[pred_fold[fold_idx] for pred_fold in models]
        best_weights=find_best_weights(predictions_from_different_models, ground_truth_labels[fold_idx], num_weights=num_weights)
        np.savetxt(os.path.join(path_to_save_weights, 'fold_%i_weights.txt' % (fold_idx + 1)), best_weights)


