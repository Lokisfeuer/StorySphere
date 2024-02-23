import random

from adventure import *
import pandas as pd


# import adventure
# from adventure import all_unit_types


def multi_for_loop(iterations):
    # a generator to iterate like multiple for loops but the amount is dynamic
    indeces = [0 for _ in iterations]
    if 0 in iterations:
        raise ValueError(f'Amount of iterations must for all instances be greater than zero. '
                         f'Received the following amounts of iterations per instance: {iterations}')
    if iterations == []:
        return
    while True:
        indeces[-1] += 1
        x = 0
        for i in range(len(indeces)):
            idx = indeces[-(i + 1)]
            max = iterations[-(i + 1)]
            x += 1
            if idx == max:
                indeces[-x] = 0
                if x == len(indeces):
                    return
                indeces[-(x + 1)] += 1
        yield indeces


'''
possible feature types:
    - float (/ Integer)
    - boolean
    - string
    - UnitId
    - list
'''


def adventure_pre_ai():
    # write a giant adventure to train the AI that will interpret lists of pre-encodings.
    # like in old_encode_adventure the generate_adventure_objs() function

    # load and prepare twitter dataset to use for texts
    twitter = pd.read_csv('twitter_data.csv')
    twitter = twitter['clean_text']
    twitter = twitter.dropna(axis=0, how='all')
    twitter = twitter.reset_index(drop=True)
    twitterator = 5  # first few entries are sometimes trash. So start at entry 5.

    # all possible values for each feature datatype.
    options = \
        {
            bool: [True, False],  # there are 2 options for boolean features
            float: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # there are infinite options for floats. I round them to 10 options
            int: list(range(10))  # there are infinite options for integers. I just take the first 10 options.
            # there are infinite options for strings. I take only one but that is always a unique tweet from Twitter.
        }
    mega_adventure = Adventure()
    for unit_type in all_unit_types:

        # get options for each feature of this unit type.
        features = unit_type().features
        specific_options = []
        for feature, feature_type in features.items():
            if feature_type in options.keys():
                specific_options.append(options[feature_type])

        # iterate over every possible combination of the options (like potenzmenge = power set)
        iterations = [len(i) for i in specific_options]
        for indeces in multi_for_loop(iterations):
            n = 0
            choice = {}  # the option taken for the unit about to be created.
            for feature, feature_type in features.items():
                if feature_type in options.keys():
                    # nth feature in the options
                    choice.update({feature: specific_options[indeces[n]]})
                    n += 1
                elif feature_type is str:
                    # take a tweet from Twitter as text.
                    choice.update({feature: twitter[twitterator]})
                    twitterator += 1
                elif feature_type is list:
                    pass  # can probably ? be ignored
                    # because elements of this adventure are intentionally created for pre-encodings
                    # and lists of UnitIds are usually irrelevant then.
                elif feature_type is UnitId:
                    pass  # can be ignored for pre-encodings
                else:
                    raise ValueError('Something went unexpectedly wrong. ')
            mega_adventure.add(unit_type(**choice))
    return mega_adventure


def all_units(adventure):
    for i in adventure.all_units.values():
        for j in i:
            yield j


def all_pre_encodings():
    all_pres = []
    adventure = adventure_pre_ai()
    for i in all_units(adventure):
        i.pre_encode()
        all_pres.append(i.pre_encoding)
    return all_pres


def many_small_adventures(n=10000):
    # generate n adventures of a realistic size. Every object has some relation to the others.
    # purpose is to train the real_encoding function with these adventures as data.
    all_adventures = []
    units = list(all_units)  # the list of all units from adventure_pre_ai
    for i in range(n):
        units_for_this_adventure = random.sample(units, random.randint(15, 30))

        # generate a list of all ids that could appear in this adventure.
        id_list = []
        adventure = Adventure()
        for unit in units_for_this_adventure:
            id_list.append(adventure.add_unit(unit))  # adventure.add_unit gives back the UnitId

        adventure = Adventure()
        for unit in units_for_this_adventure:  # iterate over every unit
            feature_values = unit.feature_values  # get feature values
            for feature, feature_type in unit.features.items():
                #  add relations between units to feature values
                if feature not in feature_values.keys():
                    if feature_type == UnitId:
                        feature_values.update({feature: random.choice(id_list)})
                    elif feature_type == tuple:  # the actual feature should in this case be a list
                        feature_values.update({feature: random.sample(id_list, random.randint(3, 10))})
                    else:
                        raise ValueError('Something went wrong unexpectedly. Please fix.')

            adventure.add_unit(unit.__class__(**feature_values))  # recreate unit with relations.
        yield adventure


def gen_real_encodings():
    for adventure in many_small_adventures():
        # generate pre encodings
        for list_of_units in adventure.all_units.values():
            for unit in list_of_units:
                unit.pre_encode()

        # get real encodings
        real_encodings = {}
        for unit_type, list_of_units in adventure.all_units.items():
            real_encodings.update({unit_type: []})
            for unit in adventure.all_units[unit_type]:
                # which version is cleaner?
                # real_encodings[unit_type].append(unit.real_encode())
                unit.real_encode()
                real_encodings[unit_type].append(unit.real_encoding)
        yield real_encodings


def all_real_encodings():
    data = {}
    for real_encodings in gen_real_encodings():
        for unit_type in real_encodings.keys():
            if unit_type not in data.keys():
                data.update({unit_type: []})
            data[unit_type].append([real_encoding for real_encoding in real_encodings[unit_type]])
    return data

