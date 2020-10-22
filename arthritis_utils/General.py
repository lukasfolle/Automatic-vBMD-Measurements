import os
import pickle


def path_exists_check(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file or path at {file_path}.")


def print_pickle(file_path):
    print(read_pickle(file_path))


def read_pickle(file_path):
    path_exists_check(file_path)
    with open(file_path, "rb") as file:
        return pickle.load(file)
