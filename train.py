from sklearn.tree import DecisionTreeClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


# Data is located at: "https://raw.githubusercontent.com/ijeendu/Machine_Learning_in_Microsoft_Azure/main/CTG.csv"

# Clean data

def clean_data(data):
    # drop rows with missing data
    x_df = data.to_pandas_dataframe().dropna()
    #drop irrelevant columns
    x_df = x_df.drop(columns=["FileName", "SegFile", "Date"])
    # extract label column
    y_df = x_df.pop("NSP")
    return x_df, y_df
    

# Create TabularDataset using TabularDatasetFactory

url = "https://raw.githubusercontent.com/ijeendu/Machine_Learning_in_Microsoft_Azure/main/CTG.csv"
ds = TabularDatasetFactory.from_delimited_files(path=url)

# clean dataset
x, y = clean_data(ds)

# Split data into train and test sets.
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size= 0.3, random_state = 0)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--max_depth', type=int, default=None, help="Maximum depth of a tree. If none, nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum number of samples required to split an internal node.")
    #parser.add_argument('--min_impurity_decrease', type=float, default=0.0, help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.")
    
    args = parser.parse_args()
    
    run.log("Maximum tree depth:", np.int(args.max_depth))
    run.log("Min samples to split:", np.int(args.min_samples_split))
    #run.log("Impurity split value:", np.float(args.min_impurity_decrease))
    
    
    model = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_split=args.min_samples_split, random_state=0).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    # insert this after fitting the model
    # create an output folder
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')
     

if __name__ == '__main__':
    main()




