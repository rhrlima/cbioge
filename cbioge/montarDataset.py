import cbioge
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import random
import pickle as pck
from sklearn.preprocessing import normalize


seed = random.randint(0, 2000)
random.seed(seed)

def mountdataset(n_folds=4):
    X = pd.read_csv("assets/data/dataset_final.csv", sep=";")
    y = pd.read_csv("assets/data/dataset_final_label.csv", sep=";")

    X.fillna(0, inplace=True) #Preenchendo os campos vazios com ZERO
    y.fillna(0, inplace=True) #Preenchendo os campos vazios com ZERO
    
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folder = 1
    
    valid_split = 0.15
    for train_index, test_index in skf.split(X, y):
        x_train, x_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        x_train, x_val, y_train, y_val  = train_test_split(x_train, y_train, test_size=valid_split, random_state=seed)

        dataset = {
            "x_train":x_train,
            "y_train":y_train,
            "x_test":x_test,
            "y_test":y_test,
            "x_valid":x_val,
            "y_valid":y_val,
            "input_shape":x_train.shape,
            "num_classes":None,
            "train_size":len(x_train),
            "test_size":len(x_test),
            "valid_size":len(x_val),
            "valid_split":valid_split
        }

        pck.dump(dataset, open(f"assets/datasets/dataset_acoes_{folder}.pickle", "wb"))
        folder+=1





if __name__ == '__main__':
    mountdataset()