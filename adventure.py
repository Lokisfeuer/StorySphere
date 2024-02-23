# this file contains the adventure class and the classes for the object types.
# Therefore, this file defines the general structure of an adventure

import this
from collections import namedtuple
import inspect
import json


class Adventure:
    def __init__(self, filename=None):  # name of adventure? I think names are more confusing than they help.
        if filename is None:
            self.all_units = {}
        else:
            self.all_units = self.load(filename)

    def __getitem__(self, unit_id):
        # this was changed recently.
        return self.all_units[unit_id['unit_type']][unit_id['nr']]

    def __len__(self):
        n = 0
        for i in self.all_units.values():
            for _ in i:
                n += 1
        return n

    # add magic methods?

    def add_unit(self, unit):
        if type(unit).__name__ not in self.all_units.keys():
            self.all_units.update({type(unit).__name__: []})
        self.all_units[type(unit).__name__].append(unit)
        unit_id = (type(unit).__name__, len(self.all_units[type(unit).__name__]))  # this line is pretty.
        return unit_id

    def save(self, filename):
        adventure = self.to_dict()
        with open(filename, 'w') as outfile:
            json.dump(adventure, outfile)

    def load(self, filename):
        # load the adventure in the file

        # read file
        with open(filename) as adventure_file:
            adventure = json.load(adventure_file)

        all_units = {}  # this function gets called before self.all_units gets initialized
        for unit_type in adventure:
            all_units.update({unit_type: []})
            for feature_values in adventure[unit_type]:
                unit_class = globals()[unit_type]  # get specific unit class
                unit = unit_class(feature_values)  # create unit object
                all_units[unit_type].append(unit)  # write all_units to how it should be
        # self.all_units = all_units  # I think this line is not nice, but I am unsure. Redundant with __init__
        return all_units

    def to_dict(self):
        # return the adventure as a json object without saving it.
        adventure = {}
        for unit_type, list_of_units in self.all_units.items():
            adventure.update({unit_type: []})
            for unit in list_of_units:
                adventure[unit_type].append(unit.to_dict())
        return adventure

    def to_vector(self, use_autoencoder=True):
        # return the vector representation of the Adventure.

        # generate pre encodings
        for list_of_units in self.all_units.values():
            for unit in list_of_units:
                unit.pre_encode()

        # get real encodings
        real_encodings = {}
        for unit_type, list_of_units in self.all_units.items():
            real_encodings.update({unit_type: []})
            for unit in self.all_units[unit_type]:
                # which version is cleaner?
                # real_encodings[unit_type].append(unit.real_encode())
                unit.real_encode()
                real_encodings[unit_type].append(unit.real_encoding)

        # TODO:
        # feed real encodings into AI
        # concat results
        if use_autoencoder:
            pass
            # feed results into AutoEncoder AI
        # return vector representation

    def to_text(self):
        raise NotImplementedError('Adventure to text doesn\'t really work yet.')


class Unit:
    def __init__(self, **feature_values):
        self.features = None
        self.set_features()
        self.check_datatypes(feature_values)
        self.feature_values = feature_values
        self.pre_encoding = None
        self.real_encoding = None

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.feature_values != other.feature_values:
            return False
        # This function ignores encodings. I am not sure whether that is good or bad.
        return True

    def to_dict(self):  # ? add "_" in front ?
        # return a dictionary version of the object.
        return self.feature_values  # I am unsure if this is perfect.

    # no load function needed

    def check_datatypes(self, feature_values):
        # features is a dict structured like this: {"feature_name": data_type_feature_should_be}.
        # if feature should be a list instead of the data_type features contains a tuple (list, element_data_type).
        for feature, val in feature_values.items():
            if feature not in list(self.features.keys()):
                raise ValueError(f'Feature "{feature}" does not exist.')

            elif isinstance(self.features[feature], tuple):
                if not isinstance(val, list):
                    raise ValueError(f'Feature "{feature}" should be of type "list".')

                if len(val) == 0:
                    pass
                elif not isinstance(val[0], self.features[feature][1]):
                    raise ValueError(f'Elements of feature "{feature}" (which is a list) '
                                     f'should be of type "{self.features[feature][1]}"')

            elif not isinstance(val, self.features[feature]):
                raise ValueError(f'Feature "{feature}" should be of type "{self.features[feature]}".')
        return

    # overwrite the following functions
    def pre_encode(self):
        # generate the pre-encoding for this object
        self.pre_encoding = None

    def real_encode(self, adventure):
        # generate the real encoding for this object.
        # get pre encodings through the adventure.
        # use AI (either Transformer or RNN) to interpret list of pres.
        self.real_encoding = None
        return self.real_encoding

    def set_features(self):
        self.features = {}
        return


# advantage of this is mostly, that now type(id) does not produce integer or string but UnitId.
UnitId = namedtuple('UnitId', 'unit_type nr')


class NotPlayerCharacter(Unit):
    def check_datatypes(self, feature_values):
        for feature, value in feature_values.items():
            if feature == 'name' and type(value) != 'string':
                raise ValueError(f"Name must be a string. Got {type(feature_values['name'])} instead.")
        return

    def set_features(self):
        self.features = {}

all_unit_types = [NotPlayerCharacter]


def write_demo_adventure():
    pass



def quicktest():
    all_globals = globals()
    global_classes = [x for x in all_globals.values() if inspect.isclass(x)]
    global_classes = global_classes[3:]
    # global_classes is now a list of all Unit Types (class objects for each)
    print(global_classes)

    id1 = UnitId('type', 3)
    print(type(id1))
    npc = NotPlayerCharacter(first=1, second=2, name='John')
    print(npc == 'thisstring')
    print(npc.real_encoding)
    x = type(npc).__name__
    print([type(npc)])


if __name__ == "__main__":
    quicktest()
