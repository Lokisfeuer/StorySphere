# this file contains the adventure class and the classes for the object types.
# Therefore, this file defines the general structure of an adventure

# import this
from collections import namedtuple
import inspect
import json
import torch
from roberta import roberta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_ENC_LENGTH = 4000
PRE_RNN_HIDDEN = 4000
PRE_SEQ_LENGTH = 50


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
                full_text += f'\tNr. {n + 1}; Name to reference this unit: "{name}"\n'
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
                    if self.features[feature][1] == UnitId:
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
                        pres.append(adventure[unit_id].pre_encoding)
                    while pres <= PRE_SEQ_LENGTH:
                        pres.append([0 for _ in PRE_ENC_LENGTH])
                    output, hidden = anna(torch.tensor(pres))
                    self.real_encoding.extend(torch.flatten(hidden).tolist())


# advantage of this is mostly, that now type(id) does not produce integer or string but UnitId.
# decide for all cases whether unit_type is string or class. Currently, its string and it should probably stay that way.
UnitId = namedtuple('UnitId', 'unit_type nr')


'''
class NotPlayerCharacter(Unit):
    # TODO: Deal with Nones for UnitIds and empty lists for lists of UnitIds
    features = {'mainname': str, 'visual description': str, 'character description': str, 'hostile': bool,
                'best friend': UnitId, 'associates': (list, UnitId), 'useless_nr': float}  # , 'useless_nr2': float
    feature_to_prompts = {
        'mainname': 'Replace this with the full name of the not-player-character.',
        'visual description': 'Replace this with the visual description of the not-player-character.',
        'character description': 'Replace this with the character description of the not-player-character.',
        'hostile': 'Replace this with "True" if the not-player-character is hostile.',
        'best friend': 'Replace this with the name of the best friend of the not-player-character which is one '
                       'of the characters from the story.',
        'associates': 'Enter a python like list of characters from the story who are associates with this character.'
    }


class Place(Unit):
    # TODO: Deal with Nones for UnitIds and empty lists for lists of UnitIds
    features = {'placename': str, 'description': str, 'places you can go from here': (list, UnitId),
                'people to be found here': (list, UnitId), 'useless_nr': float}  # , 'useless_nr2': float
    feature_to_prompts = {
        'placename': 'Replace this with a memorable name for this place.',
        'description': 'Replace this with the description of this place.',
        'places you can go from here': 'Replace this with a python like list of places you can go to from this place.',
        'people to be found here': 'Replace this with a python like list of people you can meet at this place.'
    }

'''

class EventOrScene(Unit):
    features = {
        'Which people are involved?': (list, UnitId),
        'Which groups are involved?': (list, UnitId),
        'Which beasts are involved?': (list, UnitId),
        'Which items are involved?': (list, UnitId),
        'Which secrets are involved?': (list, UnitId),
        'What motivations are involved?': (list, UnitId),
        'where might this happen?': (list, UnitId),
        'Is this an investigation scene?': bool,
        'Is this a social interaction?': bool,
        'Is this a fight scene?': bool,
        'What happens?': str,
        'How do relationships change?': str,
        'What triggers this scene to happen?': str,
        'Is this scene a start scene?': bool,
        'If this scene is a start scene, who\'s start scene is it?': (list, UnitId),
        'How likely will this scene occur?': float,
    }
    feature_to_prompts = {key: '{replace_this_with_answer}' for key in features.keys()}


class Secret(Unit):
    features = {
        'Whats the secret?': str,
        'Who knows of it?': (list, UnitId),
        'Which people are involved?': (list, UnitId),
        'Which groups are involved?': (list, UnitId),
        'Which items are involved?': (list, UnitId),
        'On a scale of 0.0 to 1.0 how exciting is it to learn about this secret?': float
    }
    feature_to_prompts = {key: '{replace_this_with_answer}' for key in features.keys()}


class Item(Unit):
    features = {
        'Who owns this?': (list, UnitId),  # technically multiple people can own something together.
        'On a scale of 0.0 to 1.0 where 0 is just very little and 1 is very much; how much is it worth?': float,
        'What is it?': str,
        'Where is it?': UnitId
    }
    feature_to_prompts = {key: '{replace_this_with_answer}' for key in features.keys()}


class Beast(Unit):
    features = {
        'Which race is this beast?': str,
        'Where could it be?': (list, UnitId),
        'What does it look like?': str,
        'On a scale of 0.0 (not aggressive) to 1.0 (immediate attack), how aggressive is the beast?': float
    }
    feature_to_prompts = {key: '{replace_this_with_answer}' for key in features.keys()}


class Group(Unit):
    features = {
        'Who is part of the group?': (list, UnitId),
        'What makes them a group / What is the reason for solidarity?': str,
        'Where did the group first meet?': UnitId
    }
    feature_to_prompts = {key: '{replace_this_with_answer}' for key in features.keys()}


class Motivation(Unit):
    '''
    Posit ive Fa ctor s
        1 ) Amb it ion
        2 ) De termina t ion
        3 ) Pa ssion
        4 ) Enthusia sm
        5 ) Cu r iosity
        6 ) Con fidence
        7 ) Op timism
        8 ) Pe r seve r a nce
        9 ) Joyful Cha llenge
        10) Gr oth
    Nega t ive Fa ctor s
        1 ) Fea r
        2 ) Fr u st r a t ion
        3 ) Anger
        4 ) Discon te n t
        5 ) Disappoin tme n t
        6 ) Dissa t isfa ct ion
        7 ) Regr e t
        8 ) Avoida nce
        9 ) Re st le ssness
        10) Despe r a t ion
    Mixe d / Ne u tr a l Fa ctor s
        1 ) Cu r iosity (posit ive or ne u tr a l)
        2 ) Re st le ssness (might le a d to posit ive change)
        3 ) Ca u tion
        4 ) Re fle ct ion
        5 ) Amb iva lence
    '''
    features = {
        'Who is motivated?': (list, UnitId),
        'What is the motivation for?': str,
        'By whom is the motivation?': (list, UnitId),
        'Is ambition the source of motivation?': bool,
        'Is determination of the source of motivation?': bool,
        'Is passion the source of motivation?': bool,
        'Is enthusiasm the source of motivation?': bool,
        'Is curiosity the source of motivation?': bool,
        'Is confidence of the source of motivation?': bool,
        'Is optimism the source of motivation?': bool,
        'Is perseverance the source of motivation?': bool,
        'Is joyful challenge the source of motivation?': bool,
        'Is growth the source of motivation?': bool,
        'Is fear the source of motivation?': bool,
        'Is frustration the source of motivation?': bool,
        'Is anger the source of motivation?': bool,
        'Is discontent the source of motivation?': bool,
        'Is disappointment the source of motivation?': bool,
        'Is dissatisfaction the source of motivation?': bool,
        'Is regret the source of motivation?': bool,
        'Is avoidance the source of motivation?': bool,
        'Is restlessness the source of motivation?': bool,
        'Is desperation the source of motivation?': bool,
        'Is caution the source of motivation?': bool,
        'Is reflection the source of motivation?': bool
    }


class Place(Unit):
    features = {
        'Where is it?': str,
        'What are the environmental conditions?': str,
        'What other places is this place associated with?': (list, UnitId),
        'What people are there?': (list, UnitId),
        'What groups are there?': (list, UnitId),
        'What beasts are there?': (list, UnitId),
        'What items are there?': (list, UnitId),
        'What secrets can be found here?': (list, UnitId),
        'What size is it? On a scale of 0.0 to 1.0 where 0 is a very small cabin and 1 is a big city.': float,
        'What does it look like?': str,
        'Whats the special history of this place?': (list, UnitId),
        'What will happen at or with this place?': str,
        'Is it a space in nature?': bool,
        'Is it an urban space?': bool,
        'Is it a desert?': bool,
        'Is it a forest?': bool,
        'Is it a mountain range?': bool,
        'Is it a body of water?': bool,
        'Is it a coastline?': bool,
        'Is it an island?': bool,
        'Is it a grassland?': bool,
        'Is it a park?': bool,
        'Is it a cave?': bool
    }


class TransportaionInfrastructure(Unit):
    features = {
        'Which places does it connect?': (list, UnitId),
        'How frequent does this route get taken? On a scale of 0.0 (barely ever) to 1.0 (constantly).': float,
        'Is this transportation infrastructure for motor vehicles?': bool,
        'Is this transportation infrastructure for non-motor vehicles?': bool,
        'Is this transportation infrastructure for pedestrians?': bool,
        'Is it a street?': bool,
        'Is it a railway?': bool,
        'Is it a flying route?': bool,
        'Is it a boat route?': bool,
        'Is it a tunnel?': bool,
        'Is it a bridge?': bool
    }


class Character(Unit):
    features = {
        'Is this a player character?': bool,
        'What is this Character skilled or talented at?': str,
        'Which Events or Scenes involve this Character?': (list, UnitId),
        'Which groups is this Character a part of?': (list, UnitId),
        'What are plans of or with this Character?': (list, UnitId),
        'What\'s this Characters backstory?': str,
        'Who is important for this Character?': (list, UnitId),
        'What is important for this Character?': (list, UnitId),
    }


all_unit_types = [Character, Place, EventOrScene, TransportaionInfrastructure, Motivation, Group, Beast, Item, Secret]


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
