from database.utils import get_reconstructed_probabilities
from models.SVM.binary_classification import get_best_estimators, get_scores_and_probs, optimise_score_over_channels
from models.scoring_metrics import scoring_function, print_scores_for_channel
from database.data import Data
from sklearn.svm import SVC
from tqdm import tqdm


train_splits = {}
train_splits['train'] = 0.7
train_splits['test'] = 0.3

#creating DATA object
ptb_binary_NN = Data(database = 'ptbdb', denoise_method='DWT', estimation_method = 'NN', train_splits=None, binary = True, parameterisation = False)

ptb_binary_NN.run()

