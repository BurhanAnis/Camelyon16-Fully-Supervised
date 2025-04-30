import os
import pickle
from tqdm import tqdm

def get_list_of_paths(dir):
    paths = []
    for path in os.listdir(dir):
        if path != ".DS_Store" and os.path.isfile(os.path.join(dir, path)):
            paths.append(dir+ '/' + path)
    return paths 

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
