import warnings
import pandas as pd
from algorithms.nearest_neighbor import NearestNeighbor

warnings.filterwarnings('ignore')

#Main method tying together the whole flow and defines file/column names
def main():
    # Read in traning data
    data_training = pd.read_csv('data/x.csv')
    data_target = pd.read_csv('data/y.csv')

    # Remove unwanted columns
    data_training_clean = data_training.drop("Unnamed: 0", axis=1)
    data_target_clean = data_target.drop("Unnamed: 0", axis=1)

    nearest_neighbor = NearestNeighbor(data_training_clean, data_target_clean)

    nearest_neighbor.execute()


#Use preprocessing.py first to create processedData.csv
if __name__ == '__main__':
    main()
