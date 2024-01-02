import pickle


# TODO: Make a better API!
# this is the API-function for the webapp.
def get_adventure(username, adventure, object_type, prefill, to_save):
    # TODO: use proper adventure structure instead of pickle.
    all_objects = setup()
    all_objects = {key[0:3].lower(): value for key, value in all_objects.items()}
    if isinstance(object_type, str):
        object_type = object_type.lower()
    with open('test.pickle', 'rb') as handle:
        full_data = pickle.load(handle)
    if username not in full_data.keys():
        user_data = {i[0:3].lower(): [] for i in all_objects.keys()}
        full_data.update({username: user_data})
    if to_save is not None:
        full_data[username][object_type].append(to_save)
        with open('test.pickle', 'wb') as handle:
            pickle.dump(full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        object_type = None
    if object_type is not None:
        if prefill is None:
            features = all_objects[object_type].features
            current_object = {i: '' for i in features.keys()}
        else:
            if isinstance(prefill, bool):
                if prefill:
                    features = all_objects[object_type].features
                    current_object = {i: 'something' for i in features.keys()}
                else:
                    features = all_objects[object_type].features
                    current_object = {i: '' for i in features.keys()}
            else:
                # this will only work when working with the real adventure because without the objects don't exist have ids'
                for i in full_data[username][object_type]:
                    break
                    if i['id'] == prefill:
                        full_data[username][object_type].remove(i)
                        current_object = i
    else:
        current_object = 'To create an object or edit one click the respective button.'
    written_adventure = f'This is the long and detailed written adventure {adventure} from user {username}.'
    listed_adventure = f'This is the long and detailed listed listed adventure {adventure} from user {username}.\n\n{full_data}'
    return current_object, listed_adventure, written_adventure

    # to_save contains an object that is to be saved to the adventure.
    if object_type is None:
        current_object = 'To create an object or edit one click the respective button.'
    else:
        # object_type is first three letters of id.
        # create current object using everything including prefill.
        current_object = None
        # if prefill is None -> make dictionary with blank values.
        # if prefill is True -> generate prefill values from AI.
        # if prefill is string -> load that object, delete it from adventure and use its features as prefill values.
        # either way current_object is a dict.
    # load adventure by username and ggf. adventure
    written_adventure = f'This is the long and detailed written adventure {adventure} from user {username}.'
    listed_adventure = f'This is the long and detailed listed listed adventure {adventure} from user {username}.'
    return current_object, listed_adventure, written_adventure


if __name__ == '__main__':
    pass
