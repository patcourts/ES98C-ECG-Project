from database.utils import get_reconstructed_probabilities
from models.SVM.binary_classification import get_best_estimators, get_scores_and_probs, optimise_score_over_channels
from models.scoring_metrics import scoring_function, print_scores_for_channel
from database.data import Data
from sklearn.ensemble import RandomForestClassifier


#define DATA object
ptb_binary_RF = Data(database = 'ptbdb', denoise_method='DWT', estimation_method = 'RF', train_splits=None, binary = True, parameterisation = True)


#define parameter grid to test
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#run data preprocessing
ptb_binary_RF.run()

#define random forest classifier 
rf = RandomForestClassifier(class_weight='balanced')

# find best hyperparameter
best_estimators_rf = get_best_estimators(ptb_binary_RF.selected_features, param_grid_rf, scoring_function, rf, ptb_binary_RF.health_state, ptb_binary_RF.nan_indices)

#get scores and probs through 3 way skfold
n_splits = 3
score_rf, bal_acc_rf, probabilities_rf, thresholds_rf, y_tests_rf, test_indices_rf = get_scores_and_probs(ptb_binary_RF.selected_features, ptb_binary_RF.health_state, ptb_binary_RF.nan_indices, best_estimators_rf, scoring_function, n_splits)

#printing scores for each channel
print_scores_for_channel(score_rf, bal_acc_rf)

#reconstructing calculated probabilities
reconstructed_probs_rf = get_reconstructed_probabilities(probabilities_rf, test_indices_rf, ptb_binary_RF.nan_indices, ptb_binary_RF.allowed_patients.count_patients(), n_splits)

#optimise scores over all channels
best_score_rf, best_channel_indices_rf = optimise_score_over_channels(reconstructed_probs_rf, thresholds_rf, ptb_binary_RF.health_state)

print(best_score_rf, best_channel_indices_rf)
