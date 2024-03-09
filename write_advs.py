import random

from adventure import *
import pandas as pd
import copy


# import adventure
# from adventure import all_unit_types


def multi_for_loop(iterations):
    # a generator to iterate like multiple for loops but the amount is dynamic
    '''
    m, n, o, ... = *iterations
    for i in range(m):
        for j in range(n):
            for k in range(o):
                ...
                    yield [i, j, k, ...]
    '''
    indeces = [0 for _ in iterations]
    if 0 in iterations:
        raise ValueError(f'Amount of iterations must for all instances be greater than zero. '
                         f'Received the following amounts of iterations per instance: {iterations}')
    if not iterations:
        return
    yield indeces
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
    # write a giant adventure to train the AI that will interpret lists of pre-encodings (Anna).
    # like in old_encode_adventure the generate_adventure_objs() function

    # load and prepare twitter dataset to use for texts
    twitter = pd.read_csv('twitter_dataset.csv')
    # twitter = pd.read_csv("twitter_data.csv")
    # twitter = twitter["clean_text"]
    twitter = twitter['Text']
    twitter = twitter.dropna(axis=0, how='all')
    twitter = twitter.reset_index(drop=True)
    twitterator = 5  # first few entries are sometimes trash. So start at entry 5.

    # all possible values for each feature datatype.
    options = \
        {
            bool: [True, False],  # there are 2 options for boolean features
            float: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # there are infinite options for floats. I round them to 10 options
            int: list(range(11))  # there are infinite options for integers. I just take the first 11 options.
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
                    choice.update({feature: specific_options[n][indeces[n]]})
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
            mega_adventure.add_unit(unit_type(**choice))

    # the next line adds one faulty unit to test the test function.
    # mega_adventure.add_unit(unit_type(npc_id1=mega_adventure.add_unit(unit_type(**choice))))
    check_mega_adventure(mega_adventure)  # just testing if everything is all right.
    check_adventure(mega_adventure)  # also just testing.
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

    # get a list of all possible units (without relations)
    units = list(all_units(adventure_pre_ai()))  # this object is mutable
    check_unit_list(units)  # checking that no Unit has an id set yet

    # iterate n times. at the end of each iteration yield one adventure
    for i in range(n):
        units_for_this_adventure = random.sample(units, random.randint(15, 30))
        check_unit_list(units_for_this_adventure)  # checking that no Unit has an id set

        # generate a list of all ids that could appear in this adventure.
        id_list = []
        adventure = Adventure()
        for unit in units_for_this_adventure:
            id_list.append(adventure.add_unit(unit))  # adventure.add_unit gives back the UnitId

        # a bit of testing that everything is all right TODO: Use assert for this part
        check_adventure(adventure)
        new_ids = []  # necessary for later tests (at the end of function)
        if len(id_list) != len(units_for_this_adventure):
            raise ValueError
        should_be_nr = 0
        for j in id_list:
            if should_be_nr != j.nr:
                raise ValueError
            should_be_nr += 1

        # write new adventure with the same units but with values for UnitIds that reference other units from the
        # adventure.
        adventure = Adventure()
        for unit in units_for_this_adventure:  # iterate over every unit
            feature_values = copy.deepcopy(unit.feature_values)  # get feature values
            # deepcopy is necessary because otherwise id_list would be mutated which is one hell of an error.

            # iterating over every feature that the unit could have (including relations)
            for feature, feature_type in unit.features.items():
                if feature not in feature_values.keys():  # check if feature is not yet set (which means its a relation)
                    #  add relations between units to feature values
                    if feature_type == UnitId:
                        feature_values.update({feature: random.choice(id_list)})
                    elif feature_type == tuple:  # the actual feature should in this case be a list of UnitIds
                        if feature_type == UnitId:
                            # this case should never happen because UnitIds are caught earlier.
                            raise ValueError("??")
                        feature_values.update({feature: random.sample(id_list, random.randint(3, 10))})
                    else:
                        # this means a feature was not set, and it did not expect a UnitId nor a list of UnitIds
                        raise ValueError('Something went wrong unexpectedly. Please fix.')

            # recreate unit with values for UnitId features and add it to adventure.
            newid = adventure.add_unit(unit.__class__(**feature_values))

            # again some checking that things are all right
            if newid not in id_list:
                raise ValueError('?')
            new_ids.append(newid)  # new_ids is also only necessary for testing.

        # and even more checking
        for unit_id in id_list:
            if unit_id not in new_ids:
                raise ValueError('?!')
        check_adventure(adventure)
        if len(adventure) != len(units_for_this_adventure):
            raise ValueError('?')

        yield adventure


# this function solemn purpose is for debugging
def check_adventure(adventure):
    for unit in all_units(adventure):
        for feature, value in unit.feature_values.items():
            if isinstance(value, UnitId):
                referenced_unit = adventure[value]  # this line might cause an error


# this function solemn purpose is for debugging as well
def check_mega_adventure(mega_adventure):
    for unit in all_units(mega_adventure):
        if 'npc_nr1' in unit.feature_values.keys():
            # this case is fine
            a = 1
            pass  # Did I mention I hate breakpoints at a pass command? They don't work.
        if 'npc_id1' in unit.feature_values.keys():
            # this case means something is wrong
            a = 1
            raise ValueError()


# this function solemn purpose is for debugging
def check_unit_list(unit_list):
    for unit in unit_list:
        if 'npc_id1' in unit.feature_values.keys():
            raise ValueError()


def gen_real_encodings(n=100):
    # generate a lot of real encodings
    # purpose is to generate training data to train Bernd rnn
    # yields a dict {unit type: list of real encodings, ...} for each adventure.

    # iterate over n small adventures
    for adventure in many_small_adventures(n=n):
        # This is similar to the to_vector function of an adventure (which can't be used before all AIs including
        # Bernd are trained)

        # generate pre encodings.
        for list_of_units in adventure.all_units.values():
            for unit in list_of_units:
                unit.pre_encode()

        # get real encodings
        real_encodings = {}
        for unit_type, list_of_units in adventure.all_units.items():
            real_encodings.update({unit_type: []})
            for unit in adventure.all_units[unit_type]:
                unit.real_encode(adventure)
                real_encodings[unit_type].append(unit.real_encoding)
        yield real_encodings


def all_real_encodings(n=100):
    data = {}
    for real_encodings in gen_real_encodings(n=n):
        for unit_type in real_encodings.keys():
            if unit_type not in data.keys():
                data.update({unit_type: []})
            data[unit_type].append([real_encoding for real_encoding in real_encodings[unit_type]])
    return data

if __name__ == "__main__":
    df = pd.read_csv("twitter_dataset.csv")
    print(df.keys())
    print(df.head())