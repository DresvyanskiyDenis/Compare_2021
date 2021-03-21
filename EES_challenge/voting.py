import os
from functools import partial

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

def save_predictions(path:str, predictions:np.ndarray, filenames:np.ndarray)->None:
    if len(predictions.shape)==1:
        predictions=predictions[..., np.newaxis]
    if len(filenames.shape)==1:
        filenames=filenames[..., np.newaxis]
    concatenated_data=np.concatenate([filenames, predictions], axis=1)
    writing_df=pd.DataFrame(data=concatenated_data, columns=['filename', 'prediction'])
    writing_df.to_csv(os.path.join(path, 'predictions.csv'), index=False)

if __name__=='__main__':
    path_to_predictions='D:\\Downloads\\best_preds_for_voting.csv'
    path_to_save='weighted_predictions'
    num_classes=3
    num_weights=1000000
    np.set_printoptions(precision=5)
    predictions_df=read_csv_file_with_predictions(path_to_predictions)
    filenames=predictions_df.iloc[:,0].values
    predictions_values=predictions_df.iloc[:,3:].values
    ground_truth_labels=predictions_df.iloc[:,1].values.reshape((-1,))
    oxanas_predictions=predictions_df.iloc[:,2].values.reshape((-1,))
    # generate weights
    weights=np.random.dirichlet(alpha=np.ones((predictions_values.shape[1],)), size=(num_weights,num_classes))
    best_UAR=0
    best_weights=None
    best_predictions=None
    major_voting_predictions=do_major_voting_to_all_instances(predictions_values)
    major_voting_UAR=recall_score(ground_truth_labels, major_voting_predictions, average='macro')
    print('Major voting UAR:%f'%major_voting_UAR)
    oxanas_UAR=recall_score(ground_truth_labels, oxanas_predictions, average='macro')
    print('Oxana\'s predictions UAR:%f'%oxanas_UAR)
    for weights_idx in range(weights.shape[0]):
        current_predictions=do_weighted_prediction_to_all_instances(predictions_values, weights[weights_idx]).reshape((-1,))
        metric=recall_score(ground_truth_labels, current_predictions, average='macro')
        if metric>best_UAR:
            print('new best weights are found. UAR:%f.Weights:%s.'%(metric,weights[weights_idx]))
            best_UAR=metric
            best_weights=weights[weights_idx]
            best_predictions=current_predictions
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            save_predictions(path_to_save, best_predictions, filenames)
            np.savetxt(os.path.join(path_to_save, 'weights.txt'), best_weights)

    print('Best UAR:%f, Best weights:%s'%(best_UAR, best_weights))
