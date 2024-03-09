# this file contains the adventure class and the classes for the object types.
# Therefore, this file defines the general structure of an adventure

import this
from collections import namedtuple
import inspect
import json
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Adventure:
    def __init__(self, filename=None):  # name of adventure? I think names are more confusing than they help.
        if filename is None:
            self.all_units = {}
        else:
            self.all_units = self._load(filename)

    def __getitem__(self, unit_id):
        return self.all_units[unit_id.unit_type][unit_id.nr]

    def __len__(self):
        n = 0
        for i in self.all_units.values():
            for _ in i:
                n += 1
        return n

    # add magic methods?

    def add_unit(self, unit) -> namedtuple:
        if type(unit).__name__ not in self.all_units.keys():
            self.all_units.update({type(unit).__name__: []})
        unit_id = UnitId(type(unit).__name__, len(self.all_units[type(unit).__name__]))
        self.all_units[type(unit).__name__].append(unit)
        return unit_id

    def save(self, filename):
        adventure = self.to_dict()
        with open(filename, 'w') as outfile:
            json.dump(adventure, outfile)

    def _load(self, filename):
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
                unit.real_encode(self)
                real_encodings[unit_type].append(unit.real_encoding)

        # iterate over the unit types.
        unit_type_encodings = []
        for unit_type, real_encoding_list in zip(self.all_units.keys(), real_encodings.values()):
            # for each unittype: feed all real encodings into Bernd AI.
            encoder = torch.load(f'bernd_encoder_{unit_type}.pt')
            _, unit_type_encoding = encoder(torch.tensor(real_encoding_list).to(device))
            # _ are all outputs, unit_type_encoding is the hidden_state (= last output)
            unit_type_encodings.append(torch.flatten(unit_type_encoding))
        # concat unit_type_encodings to one tensor.
        unit_type_encodings = torch.cat(unit_type_encodings)  # , axis=1
        if use_autoencoder:
            # TODO:
            # check if len is what AutoEncoder expects. If not, raise error.
            # this could happen when an adventure doesn't have objects of one of the types.
            # feed results into AutoEncoder AI

            pass
        return unit_type_encodings

    def to_text(self):
        raise NotImplementedError('Adventure to text doesn\'t really work yet.')

    def to_mindmap(self):
        raise NotImplementedError('Adventure to mindmap doesn\'t really work yet.')

    def to_preperation_notes(self):
        raise NotImplementedError('Adventure to preperation notes doesn\'t really work yet.')


class Unit:
    features = {}

    def __init__(self, **feature_values):
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
            # the following is similar to assert feature in list(self.features.keys())
            if feature not in list(self.features.keys()):
                raise ValueError(f'Feature "{feature}" does not exist.')

            elif isinstance(self.features[feature], tuple):
                if not isinstance(val, list):
                    raise ValueError(f'Feature "{feature}" should be of type "list". Got {type(val)} instead.')

                for i in val:
                    if not isinstance(i, self.features[feature][1]):
                        raise ValueError(f'Elements of feature "{feature}" (which is a list) should '
                                         f'be of type "{self.features[feature][1]}". Got {type(val[0])} instead.')

            elif not isinstance(val, self.features[feature]):
                raise ValueError(f'Feature "{feature}" should be of type "{self.features[feature]}".'
                                 f'Got {type(val)} instead.')
        return

    # overwrite the following functions
    def pre_encode(self):
        # generate the pre-encoding for this object
        self.pre_encoding = None
        raise NotImplementedError()

    def real_encode(self, adventure):
        # generate the real encoding for this object.
        # get pre encodings through the adventure.
        # use AI (either Transformer or RNN) to interpret list of pres.
        self.real_encoding = None
        raise NotImplementedError()


# advantage of this is mostly, that now type(id) does not produce integer or string but UnitId.
# decide for all cases whether unit_type is string or class. Currently, its string and it should probably stay that way.
UnitId = namedtuple('UnitId', 'unit_type nr')


class NotPlayerCharacter(Unit):
    features = {'npc_nr1': float, 'npc_nr2': float, 'npc_nr3': float, 'npc_id1': UnitId}

    def pre_encode(self):
        self.pre_encoding = [self.feature_values[i] for i in list(self.features.keys())[:3]]
        pass

    def real_encode(self, adventure):
        self.real_encoding = self.pre_encoding + adventure[self.feature_values['npc_id1']].pre_encoding
        # if necessary, use Anna encoder.


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
