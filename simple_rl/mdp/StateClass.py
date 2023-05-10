# Python imports
from collections.abc import Sequence

import numpy as np

''' StateClass.py: Contains the State Class. '''

class State(Sequence):
    ''' Abstract State class '''

    def __init__(self, data=[], is_terminal=False):
        self.data = data
        self._is_terminal = is_terminal

    def features(self):
        '''
        Summary
            Used by function approximators to represent the state.
            Override this method in State subclasses to have functiona
            approximators use a different set of features.
        Returns:
            (iterable)
        '''
        return np.array(self.data).flatten()

    def get_data(self):
        return self.data

    def get_num_feats(self):
        return len(self.features())

    def is_terminal(self):
        return self._is_terminal

    def set_terminal(self, is_term=True):
        self._is_terminal = is_term

    def __hash__(self):
        if type(self.data).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.data))
        elif self.data.__hash__ is None:
            return hash(tuple(self.data))
        elif isinstance(self.data, tuple):
            # if self.data contains a dict, return the hash of the first element of the tuple
            for item in self.data:
                if isinstance(item, dict):
                    if isinstance(self.data[0], dict):
                        return hash(self.data[0]['image'].tobytes())
                    else:
                        return hash(self.data[0])
            return hash(self.data)
        else:
            return hash(self.data)

    def __str__(self):
        return "s." + str(self.data)

    def __eq__(self, other):
        if isinstance(other, State):
            if isinstance(self.data, dict): # It's probably a minigrid obs
                return np.array_equal(self.data['image'], other.data['image']) and self.data['direction'] == other.data['direction']
            else:
                print(len(other.data))
                print(len(self.data))
                return self.data == other.data
        return False

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
