'''
Read csv file as pandas dataframe
'''
import pandas as pd


def print_metrics(df, labels, file_name):
    '''
    Print the metrics of the data
    Parameters:
        df: input dataframe
        labels: column containing label
        file_name: file name of the dataset
    '''
    print("File name:", file_name)
    print("Shape:", df.shape)
    labels = labels.value_counts()
    print("Number of unique labels:", len(labels))
    print("Label distribution:")
    print(labels/df.shape[0]*100)


def main():
    '''
    Entry point of the program
    '''
    files1 = ["breast.csv", "letter.csv"]
    files2 = ["rice.csv", "magic.csv"]
    files3 = ["wine.csv"]

    for file in files1:
        df = pd.read_csv(file, sep=",")
        print_metrics(df, df.iloc[:, 0], file)
        print("----------------")

    for file in files2:
        df = pd.read_csv(file, sep=",")
        print_metrics(df, df.iloc[:, -1], file)
        print("----------------")

    for file in files3:
        df = pd.read_csv(file, sep=";")
        print_metrics(df, df.iloc[:, -1], file)
        print("----------------")


if __name__ == "__main__":
    main()
