from database.utils import get_reconstructed_probabilities
from models.SVM.binary_classification import get_best_estimators, get_scores_and_probs, optimise_score_over_channels
from models.scoring_metrics import scoring_function, print_scores_for_channel
from database.data import Data
from sklearn.neighbors import KNeighborsClassifier


#define DATA object
ptb_binary_knn = Data(database = 'ptbdb', denoise_method='DWT', estimation_method = 'KNN', train_splits=None, binary = True, parameterisation = True)


#define parameter grid to test
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}


#run data preprocessing
ptb_binary_knn.run()



knn = KNeighborsClassifier()

#find best hyperparameters
best_estimators_knn = get_best_estimators(ptb_binary_knn.selected_features, param_grid_knn, scoring_function, knn, ptb_binary_knn.health_state, ptb_binary_knn.nan_indices)

#get scores and probs through 3 way skfold
n_splits = 3
score_knn, bal_acc_knn, probabilities_knn, thresholds_knn, y_tests_knn, test_indices_knn = get_scores_and_probs(ptb_binary_knn.selected_features, ptb_binary_knn.health_state, ptb_binary_knn.nan_indices, best_estimators_knn, scoring_function, n_splits)

#printing scores for each channel
print_scores_for_channel(score_knn, bal_acc_knn)

#reconstructing calculated probabilities
reconstructed_probs_knn = get_reconstructed_probabilities(probabilities_knn, test_indices_knn, ptb_binary_knn.nan_indices, ptb_binary_knn.allowed_patients.count_patients(), n_splits)

#optimise scores over all channels
best_score_knn, best_channel_indices_knn = optimise_score_over_channels(reconstructed_probs_knn, thresholds_knn, ptb_binary_knn.health_state)

print(best_score_knn, best_channel_indices_knn)
