# this file contains the adventure class and the classes for the object types.
# Therefore, this file defines the general structure of an adventure

import this
import json


class Adventure:
    def __init__(self): # name?
        self.all_objects = []
        pass

    def __getitem__(self, key):
        return self.all_objects[key]

    # add magic methods?

    def add_object(self, object_type, feature_values):
        # only use this function to create a new object or add one to the adventure.
        object_id = len(self.all_objects)
        object = None  # generate object of object type
        self.all_objects.append(object)
        return object  # is this a good idea?


    def save(self, filename):
        adventure = self.to_dict()
        with open(filename, 'w') as outfile:
            json.dump(adventure, outfile)

    def load(self, filename):
        with open(filename) as adventure_file:
            adventure = json.load(adventure_file)
        # set self attributes to what was loaded

    def to_dict(self):
        # return the adventure as a json object without saving it.
        pass

    def to_vector(self):
        # return the vector representation of the Adventure.
        for i in self.all_objects:
            i.pre_encode()
        for i in self.all_objects:
            i.real_encode(self)
        # feed real encodings into AI
        pass

    def to_text(self):
        raise NotImplementedError


class Unit:
    def __init__(self, feature_values):
        # only call this function from the adventure
        # check feature_values datatypes
        self.pre_encoding = None
        self.real_encoding = None
        pass

    def to_dict(self):  # ? add "_" in front ?
        # return a dictionary version of the object.
        pass

    # maybe add a load function

    def pre_encode(self):
        # generate the pre-encoding for this object
        self.pre_encoding = None

    def real_encode(self, adventure):
        # generate the real encoding for this object.
        # get pre encodings through the adventure.
        # use AI (either Transformer or RNN) to interpret list of pres.
        self.real_encoding = None
        pass


def write_demo_adventure():
    pass
