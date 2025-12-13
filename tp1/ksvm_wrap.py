import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import torch.nn as nn
from data import *

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1.0, param_svm_gamma="auto"):
        print(f"KSVMWrap: Starting RBF SVM training (C={param_svm_c}, Gamma={param_svm_gamma})")
        self.model = SVC(C=param_svm_c, gamma=param_svm_gamma, kernel="rbf", probability=True)
        self.model.fit(X, Y_)
        self._support_indices = self.model.support_

    def predict(self, X):
        """
        Predicts and returns class indices for data X.
        """
        return self.model.predict(X)

    def get_scores(self, X):
        """
        Returns classification scores (class probabilities) for the data.
        """
        return self.model.predict_proba(X)

    def support(self):
        """
        Property that returns the indices of support vectors.
        """
        return self._support_indices


if __name__ == "__main__":
    np.random.seed(42)

    N_COMPONENTS = 6
    N_CLASSES = 2
    N_SAMPLES_PER_COMPONENT = 10

    X_np, Y_np = sample_gmm_2d(ncomponents=N_COMPONENTS, nclasses=N_CLASSES, nsamples=N_SAMPLES_PER_COMPONENT)
    Yoh_np = class_to_onehot(Y_np)
    C_SVM = 1.0
    GAMMA_SVM = "auto"
    ksvm_model = KSVMWrap(X_np, Y_np, param_svm_c=C_SVM, param_svm_gamma=GAMMA_SVM)
    Y_ksvm = ksvm_model.predict(X_np)
    scores_ksvm = ksvm_model.get_scores(X_np)
    acc_ksvm, pr_ksvm, M_ksvm = eval_perf_multi(Y_ksvm, Y_np)
    print("\n--- KSVM (RBF) Results ---")
    print(f"Overall Accuracy: {acc_ksvm*100:.2f}%")
    print(f"Number of support vectors: {len(ksvm_model.support())}")
    print("Confusion Matrix:\n", M_ksvm)

    def decision_ksvm(X_input):
        if N_CLASSES == 2:
            return ksvm_model.model.decision_function(X_input)
        else:
            return ksvm_model.get_scores(X_input)[:, 0]

    rect = (np.min(X_np, axis=0), np.max(X_np, axis=0))
    plt.figure(figsize=(8, 6))
    offset = 0.0 if N_CLASSES == 2 else 0.5
    graph_surface(decision_ksvm, rect, offset=offset)
    graph_data(X_np, Y_np, Y_ksvm, special=ksvm_model.support())
    plt.title(f"Kernel SVM (RBF) | Acc: {acc_ksvm*100:.2f}%")
    plt.show()
