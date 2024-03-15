# this file contains the adventure class and the classes for the object types.
# Therefore, this file defines the general structure of an adventure

# import this
from collections import namedtuple
import inspect
import json
import torch
from roberta import roberta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_ENC_LENGTH = 1050
PRE_RNN_HIDDEN = 2000


class Adventure:
    def __init__(self, filename=None):  # name of adventure? I think names are more confusing than they help.
        if filename is None:
            self.all_units = {}
        else:
            self.all_units = self._load(filename)

    def __getitem__(self, unit_id):
        return self.all_units[unit_id.unit_type][unit_id.nr]

    def __setitem__(self, unit_id_to_be_set, unit):
        if not isinstance(unit_id_to_be_set, UnitId):
            if hasattr(unit_id_to_be_set, '__iter__'):
                assert len(unit_id_to_be_set) == 2
                unit_id_to_be_set = UnitId(unit_id_to_be_set[0], unit_id_to_be_set[1])
            else:
                raise ValueError
        self.all_units[unit_id_to_be_set.unit_type][unit_id_to_be_set.nr] = unit

    def __len__(self):
        n = 0
        for i in self.all_units.values():
            for _ in i:
                n += 1
        return n

    # add magic methods?
    # TODO: add __iter__ and __next__ methods to replace all_units(adventure) from write_advs.py

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
                unit = unit_class(**feature_values)  # create unit object
                all_units[unit_type].append(unit)  # write all_units to how it should be
        # self.all_units = all_units  # I think this line is not nice, but I am unsure. Redundant with __init__
        return all_units

    def all_units(self):
        for i in self.all_units.values():
            for j in i:
                yield j

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

    def to_html(self, id_to_name):
        full_text = ''
        for unit_type, unit_list in self.all_units.items():
            full_text += f'<details>'
            full_text += f'<summary>All {unit_type}, {len(unit_list)}</summary><p>'
            n = 0
            for unit in unit_list:
                full_text += f'<details>'
                name = id_to_name[UnitId(unit_type, n)]
                full_text += f'<summary>{name} Unit {unit_type} Nr. {n}</summary><p>'
                full_text += unit.to_text(id_to_name).replace('\n', '<br>')
                full_text += f'<input type=submit value="edit" onclick=\'edit("{unit_type}", "{n}");\'>'
                full_text += '</p></details>'
                n += 1
            full_text += '</p></details>'
        return full_text

    '''
    def to_html(self, id_to_name=None):
        if id_to_name is None:
            id_to_name = {}
        full_text = ''
        for unit_type, unit_list in self.to_dict().items():
            full_text += f'<details>'
            full_text += f'<summary>All {unit_type}, {len(unit_list)}</summary><p>'
            n = 0
            for unit_dict in unit_list:
                if UnitId(unit_type, n) in id_to_name.keys():
                    name = id_to_name[UnitId(unit_type, n)]
                else:
                    name = ''
                full_text += f'<details>'
                full_text += f'<summary>{name} Unit {unit_type} Nr. {unit_list.index(unit_dict)}</summary><p>'
                # TODO: use this instead: full_text += unit.to_text(id_to_name)
                for feature_name, feature_value in unit_dict.items():
                    if isinstance(feature_value, UnitId):
                        if feature_value in id_to_name.keys():
                            value = id_to_name[feature_value]
                        else:
                            value = feature_value
                    elif isinstance(feature_value, list):
                        value = []
                        for i in feature_value:
                            assert isinstance(i, UnitId)
                            if i in id_to_name.keys():
                                value.append(id_to_name[i])
                            else:
                                value.append(i)
                    else:
                        value = feature_value
                    full_text += f'{feature_name}: {value}<br>'
                full_text += f'<input type=submit value="edit" onclick=\'edit("{unit_type}", "{unit_list.index(unit_dict)}");\'>'
                full_text += '</p></details>'
                n += 1
            full_text += '</p></details>'
        return full_text
        raise NotImplementedError('Adventure to html doesn\'t really work yet')
    '''

    def to_listing(self, id_to_name):
        full_text = ''
        for unit_type, unit_list in self.all_units.items():
            full_text += f'All elements of type {unit_type}.\n'
            n = 0
            for unit in unit_list:
                name = id_to_name[UnitId(unit_type, n)]
                full_text += f'\tNr. {n + 1}: {name}\n'
                full_text += '\t\t' + '\n\t\t'.join(unit.to_text(id_to_name).splitlines())
                full_text += f'\n'
                n += 1
        return full_text


class Unit:
    features = {}

    def __init__(self, **feature_values):
        feature_values = self.check_datatypes(feature_values)
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

    def to_text(self, id_to_name):
        text = ''
        for feature_name, feature_value in self.feature_values.items():
            if isinstance(feature_value, UnitId):
                if feature_value in id_to_name.keys():
                    value = id_to_name[feature_value]
                else:
                    value = feature_value
            elif isinstance(feature_value, list):
                value = []
                for i in feature_value:
                    assert isinstance(i, UnitId)
                    if i in id_to_name.keys():
                        value.append(id_to_name[i])
                    else:
                        value.append(i)
            else:
                value = feature_value
            text += f'{feature_name}: {value}\n'
        return text

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

                n = 0
                for i in val:
                    if self.features[feature][1] == UnitId and isinstance(i, list):
                        assert hasattr(i, '__iter__')
                        assert len(i) == 2
                        i = UnitId(i[0], i[1])
                        feature_values[feature][n] = UnitId(i[0], i[1])
                    if not isinstance(i, self.features[feature][1]):
                        raise ValueError(f'Elements of feature "{feature}" (which is a list) should '
                                         f'be of type "{self.features[feature][1]}". Got {type(i)} instead.')
                    n += 1

            elif not isinstance(val, self.features[feature]):
                if self.features[feature] == UnitId:
                    assert hasattr(val, '__iter__')
                    assert len(val) == 2
                    feature_values[feature] = UnitId(val[0], val[1])
                else:
                    raise ValueError(f'Feature "{feature}" should be of type "{self.features[feature]}".'
                                     f'Got {type(val)} instead.')
        return feature_values

    def pre_encode(self):
        self.pre_encoding = []

        # get onehot of unit_type
        index = all_unit_types.index(self.__class__)
        for i in range(len(all_unit_types)):
            if i == index:
                self.pre_encoding.append(1.)
            else:
                self.pre_encoding.append(0.)

        # generate the pre-encoding for this object
        self.pre_encoding = [0. if i != index else 1. for i in range(len(all_unit_types))]
        for feature, feature_type in self.features.items():
            if feature in self.feature_values:
                value = self.feature_values[feature]
            else:
                value = None
            if feature_type == bool or feature_type == float:
                if value is not None:
                    self.pre_encoding.append(1.)
                    self.pre_encoding.append(float(value))
                else:
                    self.pre_encoding.append(0.)
                    self.pre_encoding.append(0.)
            elif feature_type == str:
                if value is not None:
                    self.pre_encoding.append(1.)
                    self.pre_encoding.extend(roberta(value))
                else:
                    self.pre_encoding.append(0.)
                    self.pre_encoding.extend([0 for _ in range(1024)])  # here needs to be roberta encoding length
            elif feature_type == UnitId:
                pass
            elif feature_type == (list, UnitId):
                pass
            else:
                raise ValueError
        assert len(self.pre_encoding) <= PRE_ENC_LENGTH
        while len(self.pre_encoding) < PRE_ENC_LENGTH:
            self.pre_encoding.append(0.)

    def real_encode(self, adventure):
        # generate the real encoding for this object.
        # get pre encodings through the adventure.
        # use AI (either Transformer or RNN) to interpret list of pres.
        self.real_encoding = self.pre_encoding
        for feature, feature_type in self.features.items():
            if feature in self.feature_values:
                value = self.feature_values[feature]
            else:
                value = None
            if feature_type == UnitId:
                if value is None:
                    self.real_encoding.append(0.)
                    self.real_encoding.extend(0. for _ in range(PRE_ENC_LENGTH))
                else:
                    self.real_encoding.append(1.)
                    self.real_encoding.extend(adventure[value].pre_encoding)
            elif feature_type == (list, UnitId):
                if value is None:
                    self.real_encoding.append(0.)
                    self.real_encoding.extend(0. for _ in range(PRE_RNN_HIDDEN))
                else:
                    self.real_encoding.append(1.)
                    # TODO: Test Anna call.
                    anna = torch.load('anna_encoder.pt')
                    pres = []
                    for unit_id in value:
                        pres.append(adventure[value].pre_encoding)
                    output, hidden = anna(torch.tensor(pres))
                    self.real_encoding.extend(hidden.to_list())


# advantage of this is mostly, that now type(id) does not produce integer or string but UnitId.
# decide for all cases whether unit_type is string or class. Currently, its string and it should probably stay that way.
UnitId = namedtuple('UnitId', 'unit_type nr')


class NotPlayerCharacter(Unit):
    # TODO: Deal with Nones for UnitIds and empty lists for lists of UnitIds
    features = {'mainname': str, 'visual description': str, 'character description': str, 'hostile': bool,
                'best friend': UnitId, 'associates': (list, UnitId)}  #

    def pre_encode(self):
        self.pre_encoding = [self.feature_values[i] for i in list(self.features.keys())[:3]]
        pass

    def real_encode(self, adventure):
        self.real_encoding = self.pre_encoding + adventure[self.feature_values['npc_id1']].pre_encoding
        # if necessary, use Anna encoder.


class Place(Unit):
    # TODO: Deal with Nones for UnitIds and empty lists for lists of UnitIds
    features = {'mainname': str, 'description': str, 'places you can go from here': (list, UnitId),
                'people to be found here': (list, UnitId)}  #

    def pre_encode(self):
        self.pre_encoding = [self.feature_values[i] for i in list(self.features.keys())[:3]]
        pass

    def real_encode(self, adventure):
        self.real_encoding = self.pre_encoding + adventure[self.feature_values['npc_id1']].pre_encoding
        # if necessary, use Anna encoder.


class EventOrScene(Unit):
    features = {
        'who is involved?': (list, UnitId),
        'where': UnitId,
        'relevant items': (list, UnitId),
        'relevant secrets': (list, UnitId),
        'Why does it happen?': str,
        'Which relationships change?': str,
        'Is this a possible startscene': bool,
        'How likely will this event happen?': float
    }


all_unit_types = [NotPlayerCharacter, Place, EventOrScene]


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
