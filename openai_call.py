import json
from openai import OpenAI
from adventure import all_unit_types, UnitId
from difflib import SequenceMatcher
import warnings


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


client = OpenAI()


def call(adventure_text, unit_type_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You receive a story in an abstract form. A story consists of units of different types. "
                           "Every unit has some features and values for these features. You will then be given a new "
                           "list of features without values but with instructions on what type of value needs to be "
                           "added to the feature. \nYou are designed to respond with a JSON object which contains "
                           "these features as keys and values for these features as values. With that a new unit can "
                           "then be created. Take care to always give fictional and innovative ideas Ideally reuse "
                           "some units from the adventure. Only adjust the "
                           "part between the brackets {} (and remove the brackets)."
            },
            {
                "role": "user",
                "content": adventure_text
            },
            {
                "role": "assistant",
                "content": "Thank you.  Now, for the new unit, please enter the features and instructions which type "
                           "is needed for which."
            },
            {
                "role": "user",
                "content": unit_type_text
            }
        ],
        temperature=1.,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # test this.
    try:
        returned_dict = json.loads(response.choices[0].message.content)  # returns dictionary
    except json.decoder.JSONDecodeError as e:
        returned_dict = {'nokey': 'No JSON given.'}
        warnings.warn(f'OpenAI did not return a proper JSON so no values can be prefilled.\n'
                      f'JSON ERROR: {e}')
    finally:
        return returned_dict

def interpret_dict(feature_string_values, unit_class, name_to_id):
    feature_values = {}
    for feature, feature_type in unit_class().features.items():
        if feature not in feature_string_values:
            # TODO: Ähnlichkeitsabfrage testen. - Works!
            similarities = []
            for feature_string in feature_string_values:
                similarities.append(similar(feature, feature_string))
            choosen_string = list(feature_string_values.keys())[similarities.index(max(similarities))]
            if not choosen_string == 'nokey':
                warnings.warn(f'Warning: A feature was not given by openai. Another features values was taken instead.\n'
                              f'Feat not given: \t"{feature}"\n'
                              f'Chosen Instead: \t"{choosen_string}"')
            # feature = choosen_string
        else:
            choosen_string = feature
        if choosen_string in feature_string_values:
            string_value = feature_string_values[choosen_string]
            if feature_type == str:
                value = string_value
            elif feature_type == float:
                if isinstance(string_value, float):
                    value = string_value
                else:
                    if string_value.isnumeric():
                        value = float(string_value)
                    else:
                        value = 0.0
            elif feature_type == bool:
                value = 'True' == string_value
            elif feature_type == UnitId:
                if string_value in name_to_id.keys():
                    value = name_to_id[string_value]
                    value = UnitId(value[0], value[1])
                else:
                    value = None  # if gpt trys to reference something not existing that gets deleted.
            elif isinstance(feature_type, tuple):
                value = []
                assert feature_type[1] == UnitId
                for val_name in string_value.split(', '):
                    if val_name in name_to_id.keys():
                        val_id = name_to_id[val_name]
                        val_id = UnitId(val_id[0], val_id[1])
                    else:
                        val_id = None
                    if val_id is not None:
                        value.append(val_id)
            else:
                # should not happen.
                raise ValueError('??')
            if value is not None:
                feature_values.update({feature: value})
    return feature_values


def ask_gpt(old_adventure, unit_type, name_to_id):
    id_to_name = {tuple(val): key for key, val in name_to_id.items()}
    adventure_text = old_adventure.to_listing(id_to_name)  # id_to_name
    unit_class = None
    for unit_class in all_unit_types:
        if unit_class.__name__ == unit_type:
            break
    assert unit_class is not None
    unit_type_text = f'A new unit of type {unit_type}.\n'
    for feature, feature_type in unit_class().features.items():
        if feature_type is bool:
            unit_type_text += f'{feature}: {{replace this this with either "True" or "False"}}'
        elif feature_type is float:
            unit_type_text += f'{feature}: {{replace this with a number between 0.0 and 1.0}}'
        elif feature_type is UnitId:
            unit_type_text += f'{feature}: {{replace this with a unit from the adventure}}'
        elif feature_type == (list, UnitId):
            unit_type_text += f'{feature}: {{replace this with one or multiple referencenames of units from the ' \
                              f'adventure. Split them with commas.}}'
        else:
            unit_type_text += f'{feature}: {{replace this with answer}}'
    unit_dict = call(adventure_text, unit_type_text)
    feature_values = interpret_dict(unit_dict, unit_class, name_to_id)
    unit = unit_class(**feature_values)
    return unit


if __name__ == '__main__':
    print(similar("What are plans of or with this Character?", "What is important for this Character?"))