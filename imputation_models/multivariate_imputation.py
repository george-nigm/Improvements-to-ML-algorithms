import miceforest as mf
from preprocessing.data_preprocessing import not_numeric_to_category


def multivariate_method_choose(X):
    # Create kernel.
    kernel = mf.KernelDataSet(X, save_all_iterations=True, random_state=0)

    kernel.mice(3)

    X = kernel.complete_data()

    return X
