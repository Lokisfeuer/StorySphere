import json
from old_encode_adventure import Adventure, all_features


def get_adventure(username, adventure, object_type, prefill, to_save):
    with open('user_to_path.json', 'r') as f:
        paths = json.load(f)
    if username not in paths.keys():
        adv = Adventure(f'First Adventure {username}')
        adv.save(f'user_written_adventures/{adv.name}')
        paths.update({username: {adv.name: f'user_written_adventures/{adv.name}'}})
    if adventure is not None:
        if adventure not in paths[username]:
            adv = Adventure(f'{adventure} {username}')
            adv.save(f'user_written_adventures/{adv.name}')
            paths[username].update({adventure: f'user_written_adventures/{adv.name}'})
    else:
        adventure = f'First Adventure {username}'
    adv = Adventure(adventure)
    adv.load(f'{paths[username][adventure]}')  # adventure=None works fine here.
    current_object = ''
    if object_type is not None:
        name_to_feat = {'spi': adv.sci, 'mot': adv.mot, 'eve': adv.eus, 'npc': adv.npc, 'geh': adv.geh, 'gru': adv.gru,
                        'bea': adv.bea, 'geg': adv.geg}
        cla = name_to_feat[object_type]
        if to_save is not None:
            objs = all_features[cla.name]
            vals = {}
            for feat, val in to_save.items():
                if feat in objs['texts']:
                    vals.update({feat: val})
                elif feat in objs['single_ids']:
                    # check for ids
                    vals.update({feat: val})
                elif feat in objs['bools']:
                    if val.lower().startswith(('y', 'yes', 'yup', 'yeah', '1')):
                        vals.update({feat: True})
                    else:
                        vals.update({feat: False})
                elif feat in objs['scalars']:
                    try:
                        vals.update({feat: float(val)})
                    except ValueError:
                        vals.update({feat: 0.0})
                elif feat in objs['list_ids']:
                    vals.update({feat: val.split(', ')})
                else:
                    raise ValueError
            cla.add(**vals)
            # maybe change datatypes here and there.
        else:
            if isinstance(prefill, str):
                pass  # TODO edit object
            elif prefill is None:
                pass  # This case only comes to play with the / route.
            elif prefill:
                current_object = {}
                for i in cla.features.keys():
                    objs = all_features[cla.name]
                    if i in objs['texts']:
                        current_object.update({i: '"This is awesome and definitely AI-generated text!"'})
                    elif i in objs['single_ids']:
                        current_object.update({i: 'id_spi_1'})
                    elif i in objs['bools']:
                        current_object.update({i: 'True'})
                    elif i in objs['scalars']:
                        current_object.update({i: '1.0'})
                    elif i in objs['list_ids']:
                        current_object.update({i: 'id_spi_1, id_spi_2, id_spi_3'})
                    else:
                        raise ValueError
            else:
                current_object = {}
                for i in cla.features.keys():
                    objs = all_features[cla.name]
                    if i in objs['texts']:
                        current_object.update({i: '"some text here"'})
                    elif i in objs['single_ids']:
                        current_object.update({i: 'id_obj_nr'})
                    elif i in objs['bools']:
                        current_object.update({i: 'False'})
                    elif i in objs['scalars']:
                        current_object.update({i: '0.0'})
                    elif i in objs['list_ids']:
                        current_object.update({i: '"id_obj_nr, id_obj_nr, ..."'})
                    else:
                        raise ValueError
    adv.save(f'{paths[username][adventure]}')
    with open('user_to_path.json', 'w') as f:
        f.write(json.dumps(paths, indent=4))
    return current_object, adv.to_list(), adv.to_text()


if __name__ == '__main__':
    pass
