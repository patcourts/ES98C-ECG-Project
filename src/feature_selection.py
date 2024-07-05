from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, mutual_info_classif #for forward feature selection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scoring_metrics import scoring_function
from utils import reduce_parameter_set, reduce_parameter_set_single, convert_single_dict_to_array


def forward_selection(params, health_state, scorer=scoring_function, k=6):
    
    X_train, X_test, y_train, y_test = train_test_split(params, health_state, test_size=0.3, stratify = health_state)

    #initialise svc
    svm = SVC(class_weight='balanced')

    SFS_forward = SequentialFeatureSelector(estimator=svm, scoring=scoring_function, cv=3, n_features_to_select=k)

    SFS_forward.fit(X_train, y_train)

    selected_indices = SFS_forward.get_support(indices=True)
    
    return selected_indices


def filter_method(params, health_state, k=6, scorer=mutual_info_classif):
    # Apply SelectKBest with mutual information
    selector = SelectKBest(score_func=scorer, k=k)
    
    
    X_new = selector.fit_transform(params, health_state)
    
    # Get the indices of selected features
    selected_indices = selector.get_support(indices=True)
    
    return selected_indices


def select_features(params, params_array, health_state, nan_indices, desired_no_feats):
    selected_params = {}
    print(f'selecting {desired_no_feats} most important features')
    for i in range(0, len(nan_indices)):
        #find indices to keep through filter method
        
        chosen_indices_filt = filter_method(params_array[i], health_state[nan_indices[i]], k=8)
        
        #reduce the parameter based on these indices
        reduced_params = reduce_parameter_set(params, chosen_indices_filt, i)

        #convert new params dict back into array
        reduced_params_array, no_features = convert_single_dict_to_array(reduced_params, health_state[nan_indices[i]])

        #find 4 best features through sfs
        chosen_indices_filt_sfs = forward_selection(reduced_params_array, health_state[nan_indices[i]], scorer=scoring_function, k=desired_no_feats)

        selected_params[i] = reduce_parameter_set_single(reduced_params, chosen_indices_filt_sfs)

    return selected_params