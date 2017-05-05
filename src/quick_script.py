from data_preprocessing import *
from utility import *
from visualization import *

(X_train, Y_train), (X_test, Y_test) = load_data()
predictions = np.genfromtxt(RESULTS_DIR + "predictions.txt")

visualize_pr_curves(Y_test, predictions)
