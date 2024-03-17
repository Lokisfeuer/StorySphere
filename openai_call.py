import json
from openai import OpenAI
from adventure import all_unit_types, UnitId
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
                           "then be created. Take care to always give fictional and innovative ideas."
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
        temperature=1.35,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return json.loads(response.choices[0].message.content)  # returns dictionary


def interpret_dict(feature_string_values, unit_class, name_to_id):
    feature_values = {}
    for feature, feature_type in unit_class().features.items():
        if feature in feature_string_values:
            string_value = feature_string_values[feature]
            if feature_type == str:
                value = string_value
            elif feature_type == float:
                if string_value == '':
                    value = 0.0
                else:
                    value = float(string_value)
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
                for val_name in string_value:
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
    id_to_name = {val: key for  key, val in name_to_id.items()}
    adventure_text = old_adventure.to_listing(id_to_name)  # id_to_name
    unit_class = None
    for unit_class in all_unit_types:
        if unit_class.__name__ == unit_type:
            break
    assert unit_class is not None
    unit_type_text = f'A new unit of type {unit_type}.\n'
    for feature, prompt in unit_class().feature_to_prompts.items():
        unit_type_text += f'{feature}: {{{prompt}}}\n'
    unit_dict = call(adventure_text, unit_type_text)
    feature_values = interpret_dict(unit_dict, unit_class, name_to_id)
    unit = unit_class(**feature_values)
    return unit

