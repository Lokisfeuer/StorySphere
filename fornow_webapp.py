# !Warning! This program is not safe. Also it is very instable.
# TODO: Add user authentification (not for prototype)
# TODO: Stop using cookies - it is terrible. (not for prototype)
# TODO: Implement to_html method for adventure  //
# TODO: add lists of UnitIds (and everything else) when writing objects //
# TODO: add possibility to edit objects. //
# TODO: test properly
from flask import *
from adventure import Adventure, NotPlayerCharacter, all_unit_types, UnitId
import os
import random
import string
from markupsafe import Markup
from write_advs import all_units

app = Flask(__name__)

# TODO Make this an environment variable
# TODO Make this a constant! (Don't upload it on github)
app.secret_key = os.urandom(24)


@app.route("/about_us")
def about_us():
    return render_template('about_us.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        session['password'] = '42'
        session['adventure name'] = request.form['adventure_name']
        if request.form['adventure_name'] == '':
            session['adventure name'] = 'first adventure'
        session['object_type'] = None
        for i in ['name_to_id', 'unsetname_to_list_of_setnames']:
            if i not in session:
                session[i] = {}
        if 'unset_names' not in session:
            session['unset_names'] = []
        return redirect(url_for('index'))
    rand_str = ''.join(random.choice(string.ascii_letters) for x in range(12))

    return f'''
        <form method="post">
            <p>Please enter your username:<br><input type=text name=username value="{rand_str}">
            <p>You can also specify the adventure name if you want to:<br><input type=text name=adventure_name>
            <p><input type=submit value=Login>
        </form>
    '''


@app.route('/logout')
def logout():
    # TODO: delete full session NO DONT! But think about what makes sense to delete here and what does not.
    # remove the username from the session if it's there
    session.pop('username', None)
    session.pop('password', None)
    return 'You logged out.'


@app.route("/hello_world")
def hello_world():
    return "<h1>Hello World</h1>"


@app.route("/mindmap", methods=["GET"])
def mindmap():
    if 'username' not in session:
        return redirect(url_for('login'))
    raise NotImplementedError


@app.route('/write_object', methods=['POST'])
def write_object():
    assert 'new object' in request.form.keys()
    # This means the user pressed on "new object" or on "edit".
    # This function need to return the formular to write an object of the type.

    unit_type = request.form['new object']  # get value of pressed button which is "new {unit_type}".
    assert unit_type is not None
    # remove the "new" so should give a unit type string like "NotPlayerCharacter"
    unit_type = '_'.join(unit_type.split()[1:])
    # if user wants to edit an object unit_type will now end in the number of said object.
    unit_type, unit_nr = get_unit_nr(unit_type)

    # create unit to get default values in the form.
    # If user is editing unit name_default is its name. Otherwise, it's None.
    unit, name_default = get_unit(unit_type, unit_nr)
    name = write_name_select_box(name_default)

    # write the pre-filled form with the values from the unit.
    unit_text = write_pre_filled_form(unit)

    unit_text = \
        f'<form name="form" method="POST" action="/save_unit">' \
        f'{name}{unit_text}' \
        f'<input type=submit id="submitbutton" name="save {unit_type}" value="save {unit_type}" disabled></form>' \
        f'<br>(if button does not enable, change name to reference, then change it back and try again.)'
    unit_text = Markup(unit_text)
    return render_template('edit_object.html', object_text=unit_text)


@app.route('/save_unit', methods=['POST'])
def save_unit():
    # TODO: Comment through this function
    # The user just submitted the form to create a new object.
    # This function needs to create the object, append it to the adventure and return a redirect to index.

    # this means the save button was just pressed.
    # add object to adventure and save adventure
    # save what user entered as ideal output (with former adventure) to train the AI
    # if objects that don't exist were mentioned by the user, put them on a list of objects that need to be
    # added to the adventure.
    name = request.form['name']

    # get one demo unit to get features.
    unit = None
    # get unit_class
    for i, j in request.form.items():
        if i.startswith('save '):
            unit_type = i[5:]  # the button has value 'save {unit_type}'.
            for unit_class in all_unit_types:
                if unit_class.__name__ == unit_type:
                    unit = unit_class()
                    break
            break
    assert unit is not None

    # get feature_values
    feature_values = {}
    for feature, feature_type in unit.features.items():
        if feature in request.form.keys():
            if feature_type == str:
                value = request.form[feature]
            elif feature_type == float:
                if request.form[feature] == '':
                    value = 0.0
                else:
                    value = float(request.form[feature])
            elif feature_type == bool:
                value = 'True' == request.form[feature]
                assert isinstance(value, bool)
            elif feature_type == UnitId:
                value = register_name(request.form[feature], feature, selfname=name)
            elif isinstance(feature_type, tuple):
                value = []
                assert feature_type[1] == UnitId
                for val_name in request.form.getlist(feature):
                    val_id = register_name(val_name, feature, selfname=name)
                    if val_id is not None:
                        value.append(val_id)
            else:
                # should not happen.
                raise ValueError('??')
            if value is not None:
                feature_values.update({feature: value})

    unit = unit_class(**feature_values)

    # add to adventure
    adventure = load_or_save_adventure()
    if name in session['name_to_id'].keys():
        # this means an already existing object was edited
        unit_id = session['name_to_id'][name]
        adventure[unit_id] = unit
    else:
        # TODO: save adventure and unit to training data
        unit_id = adventure.add_unit(unit)

    # check names and ids.
    name = request.form['name']
    if name in session['unset_names']:
        session['unset_names'].remove(name)
    session['name_to_id'].update({name: unit_id})

    # iterate over already existing units that reference the just created one.
    if name in session['unsetname_to_list_of_setnames'].keys():
        for i in session['unsetname_to_list_of_setnames'][name]:
            # i is a tuple: (name, the feature where the reference is)
            i_unit_id = session['name_to_id'][i[0]]
            i_unit_id = UnitId(i_unit_id[0], i_unit_id[1])

            # updating unit without changing any ids.
            unit = adventure[i_unit_id]
            feature_values = unit.feature_values
            if unit.features[i[1]] == UnitId:
                feature_values.update({i[1]: unit_id})
            elif unit.features[i[1]] == (list, UnitId):
                if i[1] in feature_values.keys():
                    feature_values[i[1]].append(unit_id)
                else:
                    feature_values.update({i[1]: [unit_id]})
            unit = unit.__class__(**feature_values)
            adventure[i_unit_id] = unit

    load_or_save_adventure(adventure=adventure)

    session.modified = True

    return redirect(url_for('index'))


def register_name(name, feature, selfname):
    if name in session['name_to_id']:
        value = session['name_to_id'][name]
        return UnitId(value[0], value[1])
        # the unitId class gets transformed to a normal tuple in the session.
        # So it needs to be remade here.
    if name not in session['unset_names']:
        session['unset_names'].append(name)
        session['unsetname_to_list_of_setnames'].update({name: []})
    session['unsetname_to_list_of_setnames'][name].append((selfname, feature))


@app.route('/')
def index():
    # this function displays the adventure currently being written in good human-readable format.
    if 'username' not in session:
        return redirect(url_for('login'))
    adventure = load_or_save_adventure()  # load adventure
    id_to_name = {val: key for key, val in session['name_to_id'].items()}
    in_html = adventure.to_html(id_to_name)
    # in_html = json.dumps(adventure.to_dict(), indent=4)
    # in_html = f'This is a placeholder for the actual adventure which has a length of {len(adventure)}.'
    in_html = Markup(f'{in_html}')
    return render_template('display_adventure.html', adventure=in_html)


def get_unit_nr(unit_type):
    # if unit_type ends on a numeral split unit_type in the actual string and the number following it.
    unit_nr = None
    if unit_type[-1].isnumeric():
        n = 1
        while unit_type[-1 * n].isnumeric():
            n += 1
        n = n - 1  # you think this is dumb bullshit going in circles? You might be correct.
        unit_nr = int(unit_type[-1 * n:])
        unit_type = unit_type[:-1 * n]
    return unit_type, unit_nr


def call_ai(adventure, unit_type):
    # TODO add AI call here. AI should return something like the following object.
    return NotPlayerCharacter(npc_nr1=0.1, npc_nr2=0.2, npc_nr3=0.3, npc_bool1=True)


def get_unit(unit_type, unit_nr):
    adventure = load_or_save_adventure()
    if unit_nr is None:
        # get object from AI
        unit = call_ai(adventure, unit_type)
        default_name = None  # the default name to reference this object. Needs to be user-given.
    else:
        # load object from adventure
        unit_id = UnitId(unit_type, unit_nr)
        unit = adventure[unit_id]
        id_to_name = {val: key for key, val in session['name_to_id'].items()}
        assert unit_id in id_to_name.keys()
        default_name = id_to_name[unit_id]
    return unit, default_name


def write_name_select_box(default_name):
    name = ''
    if default_name is not None:
        name += f'<option selected="selected">{default_name}</option>'  # ?maybe disable selectbox here.
    for i in session['unset_names']:
        name += f'<option>{i}</option>'
    # if name_default is None:
    name = f'Name to reference this object: <select name="name" class="js-single-choice" onchange="checkform();">{name}</select><br>'
    return name


def write_pre_filled_form(unit):
    # TODO: clean this function and comment through it.
    # object_text = f'Unit Type: {unit_type}.<br>'
    unit_text = ''
    for feature, feature_type in unit.features.items():
        if feature in unit.feature_values:
            value = unit.feature_values[feature]
        else:
            value = None
        if feature_type == float:
            if value is None:
                value = 0.0
            unit_text += f'{feature}: <input type="number" step="0.01" name="{feature}" value="{value}"><br>'
        elif feature_type == str:
            if value is None:
                value = ''
            unit_text += f'{feature}: <input type=text name="{feature}" value="{value}"><br>'
        elif feature_type == bool:
            if value is None:
                value = False
            if value:
                unit_text += f'{feature}: <input type=checkbox name="{feature}" value="True"><br>'
            else:
                unit_text += f'{feature}: <input type=checkbox name="{feature}" value="True" checked><br>'
        elif isinstance(feature_type, tuple) or feature_type == UnitId:
            id_to_name = {val: key for key, val in session['name_to_id'].items()}
            if value in id_to_name.keys():
                value = id_to_name[value]
            all_names = session['unset_names'] + list(session['name_to_id'].keys())
            option_text = '<option></option>'
            for i in all_names:
                if i == value:
                    option_text += f'<option selected="selected">{i}</option>'
                else:
                    option_text += f'<option>{i}</option>'
            if isinstance(feature_type, tuple):
                unit_text += f'{feature}: <select name="{feature}" class="js-multi-choice-with-clear" multiple="multiple">' \
                             f'{option_text}</select><br>'
            else:
                unit_text += f'{feature}: <select name="{feature}" class="js-single-choice-with-clear">' \
                             f'{option_text}</select><br>'
        else:
            # should not happen
            raise ValueError('??')
    return unit_text


def load_or_save_adventure(adventure=None):
    username = session['username']
    adventure_name = session['adventure name']
    filename = f'user_written_adventures/{username}/{adventure_name}.json'
    if not os.path.exists(f'user_written_adventures/{username}'):
        os.mkdir(f'user_written_adventures/{username}')
        if adventure is None:
            adventure = Adventure()
        adventure.save(filename=filename)
        return adventure
    if adventure is not None:
        adventure.save(filename=filename)
        return adventure
    adventure = Adventure(filename=filename)
    return adventure


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
