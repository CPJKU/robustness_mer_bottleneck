import pickle


def pickle_load(fp):
    """ Loads pickled file and returns it. """
    with open(fp, 'rb') as f:
        return pickle.load(f)


def pickle_save(fp, item):
    """ Saves (pickle) file. """
    with open(fp, 'wb') as f:
        pickle.dump(item, f)
