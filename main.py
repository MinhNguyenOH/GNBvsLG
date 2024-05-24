'''
Libraries for Gaussian Naive Bayes, Logistic Regression, plotting graph, etc.
'''
import sys
import diffprivlib.models as dp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def plot_accuracy_graph(epsilons, nonprivate_accuracy, accuracy, model_name):
    '''
    Plot a graph illustrating how model's accuracy changes corresponding
    to changes in the value of epsilon
    Parameters:
        epsilons: a list of epsilon values
        nonprivate_accuracy: a list of accuracy values of the non-private model
        accuracy: a list of accuracy values of the differentially private model
        model_name: name of the model
    Returns:
        None
    '''
    plt.plot(epsilons, nonprivate_accuracy, linestyle='dashed', color='red')
    plt.plot(epsilons, accuracy)
    plt.xscale('log')
    plt.title(f"Differentially private {model_name} accuracy")
    plt.xlabel("epsilon")
    plt.ylabel("Accuracy")
    plt.show()


def get_non_private_max_iter(filename):
    '''
    Get the maximum number of iterations to train the non-private
    Logistic Regression model. Different datasets converge after
    different number of iterations. Tuning this parameters to make
    sure the model converges
    Parameters:
        filename: name of the dataset for training model
    Returns:
        maximum number of iterations corresponding to a given dataset
    '''
    if filename == "breast.csv":
        return 2000
    elif filename == "wine.csv":
        return 6000
    elif filename == "rice.csv":
        return 100
    elif filename == "letter.csv":
        return 3000
    elif filename == "magic.csv":
        return 200

    return 100


def get_private_max_iter(filename):
    '''
    Get the maximum number of iterations to train the differentially
    private Logistic Regression model. Different datasets converge
    after different number of iterations. Tuning this parameters to
    make sure the model converges
    Parameters:
        filename: name of the dataset for training model
    Returns:
        maximum number of iterations corresponding to a given dataset
    '''
    if filename == "breast.csv":
        return 200
    elif filename == "wine.csv":
        return 300
    elif filename == "rice.csv":
        return 300
    elif filename == "letter.csv":
        return 200
    elif filename == "magic.csv":
        return 500

    return 100


def load_dataset(filename, target_col, num_cols):
    '''
    Load the given dataset into numpy arrays
    Parameters:
        filename: name of the dataset used for training model
        target_col: column index of the label column
        num_cols: number of columns in the dataset
    Returns:
        a tuple of two numpy arrays (X, y) where X contains 
        feature values and y contains target (label) values
    '''
    cols = set([i for i in range(num_cols)])
    cols.remove(target_col)

    X = np.loadtxt(f"data/{filename}", usecols=cols, delimiter=",")
    y = np.loadtxt(f"data/{filename}", usecols=target_col,
                   dtype=str, delimiter=",")

    return (X, y)


def evaluate_gaussian_naive_bayes(filename, target_col, num_cols):
    '''
    Evaluate the accuracy of the differentially private Gaussian Naive
    Bayes classification model
    Parameters:
        filename: name of the dataset used for training model
        target_col: column index of the label column
        num_cols: number of columns in the dataset
    Returns:
        None
    '''
    X, y = load_dataset(filename, target_col, num_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Naive Bayes with no privacy
    nonprivate_clf = GaussianNB()
    nonprivate_clf.fit(X_train, y_train)
    nonprivate_score = nonprivate_clf.score(X_test, y_test)
    nonprivate_accuracy = [nonprivate_score for _ in range(50)]

    # Differentially private Gaussian Naive Bayes
    # Specify a default bounds to prevent the possibility of privacy leakage
    bounds = (-1e5, 1e5)
    epsilons = np.logspace(-2, 2, 50)
    accuracy = list()

    for epsilon in epsilons:
        clf = dp.GaussianNB(epsilon=epsilon, bounds=bounds)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        accuracy.append(score)

    plot_accuracy_graph(epsilons, nonprivate_accuracy,
                        accuracy, "Gaussian Naive Bayes")


def evaluate_logistic_regression(filename, target_col, num_cols):
    '''
    Evaluate the accuracy of the differentially private Logistic Regression
    classification model
    Parameters:
        filename: name of the dataset to train model
        target_col: column index of the label column
        num_cols: number of columns in the dataset
    Returns:
        None
    '''
    X, y = load_dataset(filename, target_col, num_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    max_iter = get_non_private_max_iter(filename)

    # Logistic Regression with no privacy
    nonprivate_clf = LogisticRegression(solver="lbfgs", max_iter=max_iter)
    nonprivate_clf.fit(X_train, y_train)
    nonprivate_score = nonprivate_clf.score(X_test, y_test)
    nonprivate_accuracy = [nonprivate_score for _ in range(50)]

    # Differentially private Logistic Regression
    epsilons = np.logspace(-2, 2, 50)
    accuracy = list()
    # Specify the max l2 norm of data to prevent the possibility of privacy leakage
    max_l2_norm = np.max(np.linalg.norm(X_train, axis=1))
    max_iter = get_private_max_iter(filename)

    for epsilon in epsilons:
        clf = dp.LogisticRegression(
            epsilon=epsilon, data_norm=max_l2_norm, max_iter=max_iter)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        accuracy.append(score)

    plot_accuracy_graph(epsilons, nonprivate_accuracy,
                        accuracy, "Logistic Regression")


def main():
    '''
    Entry point of the program
    '''
    if len(sys.argv) != 5:
        print("Invalid number of command-line arguments")
        print(
            f"Usage: python3 {sys.argv[0]} <model> <dataset> <target_col> <num_cols>")
        sys.exit(1)

    model = sys.argv[1]
    dataset = sys.argv[2]
    target_col = int(sys.argv[3])
    num_cols = int(sys.argv[4])

    if model == "gnb":
        evaluate_gaussian_naive_bayes(dataset, target_col, num_cols)
    elif model == "logit":
        evaluate_logistic_regression(dataset, target_col, num_cols)


if __name__ == "__main__":
    main()
