import pickle as pkl


def saver(filename, data):
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

def loader(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)