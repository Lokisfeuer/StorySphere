# import old_encode_adventure
import random
import adventure


def adventure_pre_ai():
    # write the giant adventure like in old_encode_adventure the generate_adventure_objs() function
    adventure = None
    return adventure


def many_random(giant_adventure):  # TODO: find better name for giant_adventure
    advs = []
    for i in range(10000):
        adv = encode_adventure.Adventure(f'DataAdventure{i}')
        name_to_feat = {'sci': adv.sci, 'mot': adv.mot, 'eus': adv.eus, 'npc': adv.npc, 'geh': adv.geh,
                        'gru': adv.gru, 'bea': adv.bea, 'geg': adv.geg}
        this = random.sample(all_ids, random.randint(15, 30))
        for i in this:
            f_v = add_references(giant_adv, i, [j for j in this if j != i])
            _, feat, nr = tuple(i.split('_'))
            name_to_feat[feat].add(**f_v)
        advs.append(adv)
        pass



all_features = {
    'sci': {
        'texts': ['name', 'backstory'],
        'bools': ['charakterbogen', 'plaene_fuer_den_charakter', 'hat_eine_backstory'],
        'scalars': [],
        'single_ids': [],
        'list_ids': ['startszene', 'events', 'gruppen', 'backstory_sonstiges']
    },
    'eus': {
        'texts': ['was', 'warum'],
        'bools': ['untersuchen', 'soziale_interaktion', 'fight', 'start'],
        'scalars': ['schwierigkeitsgrad', 'wahrscheinlichkeit'],
        'single_ids': [],
        'list_ids': ['wer', 'wo', 'Gegenstände', 'Geheimnisse', 'personen', 'wer_muss_da_sein', 'wo_kann_das_sein',
                     'motivationen']
    },
    'npc': {
        'texts': ['name', 'backstory'],
        'bools': ['charakterbogen', 'plaene', 'hat_eine_backstory'],
        'scalars': [],
        'single_ids': [],
        'list_ids': ['events_und_szenen', 'gruppen', 'backstory_sonstiges']
    },
    'geh': {
        'texts': ['was'],
        'bools': [],
        'scalars': ['positivitaet'],
        'single_ids': [],
        'list_ids': ['wer_weiss_davon', 'wen_und_was_betrifft_das']
    },
    'gru': {
        'texts': ['grund_des_zusammenhalts'],
        'bools': [],
        'scalars': [],
        'single_ids': ['moegliche_motivation_von_aussen', 'geburtsort_der_gruppe'],
        'list_ids': []
    },
    'geg': {
        'texts': ['was'],
        'bools': [],
        'scalars': ['wert'],
        'single_ids': [],
        'list_ids': ['wessen', 'wo']
    }
}


def get_f_v_by_id(id, adv):
    name_to_feat = {'sci': adv.sci, 'mot': adv.mot, 'eus': adv.eus, 'npc': adv.npc, 'geh': adv.geh, 'gru': adv.gru,
                    'bea': adv.bea, 'geg': adv.geg}
    _, feat, nr = tuple(id.split('_'))
    feat = name_to_feat[feat]
    f_v = feat.all_objects[nr]
    del f_v['ID']
    return f_v


def add_references(giant_adv, main_id, other_ids):
    _, feat, nr = tuple(main_id.split('_'))
    f_v = get_f_v_by_id(main_id, giant_adv)
    feats = all_features[feat]
    for i in feats['single_ids']:
        f_v[i] = random.choice(other_ids)
    for i in feats['list_ids']:
        f_v[i] = random.sample(other_ids, random.randint(3, 10))
    return f_v
