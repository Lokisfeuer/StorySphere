from flask import Flask, request, render_template, session, redirect, url_for
from markupsafe import Markup
import old_main as main
import os

app = Flask(__name__)

# TODO Make this an environment variable
# TODO Make this a constant! (Don't upload it on github)
app.secret_key = os.urandom(24)


def return_template(session, prefill=None, to_save=None):
    if prefill is not None and to_save is not None:
        raise ValueError
    if 'username' not in session:
        return redirect(url_for('login'))
    if 'adventure' in session:
        adventure = session['adventure']
    else:
        adventure = None
    if 'object_type' in session:
        object_type = session['object_type']
    else:
        object_type = None
    current_object, listed_adventure, written_adventure = main.get_adventure(
        session['username'], adventure, object_type, prefill, to_save
    )
    if isinstance(current_object, dict):
        com = ''
        for i, j in current_object.items():
            com += f'{i} <input type=text name={i} value={j}><br>'
        com = f'''<form method="post" action="/save">{com}<input type=submit value=save></form>'''
        current_object = Markup(com)
    return render_template(
        'main.html',
        current_object=current_object,
        listed_adventure=listed_adventure,
        written_adventure=written_adventure
    )


@app.route('/')
def index():
    '''
    <script>
        var myHyperVariable;
        myHyperVariable="{variable}";
    </script>
    <script>
        function myFunction() {
            document.getElementById("123").value = myHyperVariable;
        }
    </script>
    <input type="text" id="123" name="first_name">
    <button onclick="myFunction()">The Click</button>'''
    variable = 'Hi'
    variable = f'''<script>
    var myHyperVariable;
    myHyperVariable="{variable}";
</script>
'''
    variable = Markup(variable)
    # val = f'<script>var myHyperVariable; myHyperVariable="{variable}";</script>'
    return render_template('main.html', variable=variable)
    # if 'username' not in session:
    #    return redirect(url_for('login'))
    # current_object = Markup('''<p><input type=text name=username>''')
    return return_template(session)


@app.route('/save', methods=['POST'])
def save():
    obj = {}
    for i in request.form:
        obj.update({i: request.form[i]})
    return return_template(session, to_save=obj)


@app.route('/edit', methods=['POST'])
def edit():
    id = request.form['object_id']
    session['object_type'] = id[0:3]
    return return_template(session, prefill=id)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        session['password'] = '42'
        return redirect(url_for('index'))
    return '''
        Please enter your username:
        <br>
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''


@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    session.pop('adventure', None)
    session.pop('password', None)
    session.pop('object_type', None)
    return 'You logged out.'


@app.route('/choose_adventure', methods=['GET', 'POST'])
def choose_adventure():
    if request.method == 'POST':
        session['adventure'] = request.form['adventure']
        return redirect(url_for('index'))
    return '''
        Please enter the name of the adventure you wish to write or continue writing:
        <br>
        <form method="post">
            <p><input type=text name=adventure>
            <p><input type=submit value=Login>
        </form>
    '''


# here come the baby-functions
# these functions should set object_type in the session.
# These functions return_template with or without prefill=True.
@app.route('/filled_spi', methods=['POST'])
def filled_spi():
    session['object_type'] = 'spi'
    return return_template(session, prefill=True)


@app.route('/blank_spi', methods=['POST'])
def blank_spi():
    session['object_type'] = 'spi'
    return return_template(session, prefill=False)


@app.route('/filled_mot', methods=['POST'])
def filled_mot():
    session['object_type'] = 'mot'
    return return_template(session, prefill=True)


@app.route('/blank_mot', methods=['POST'])
def blank_mot():
    session['object_type'] = 'mot'
    return return_template(session, prefill=False)


@app.route('/filled_eve', methods=['POST'])
def filled_eve():
    session['object_type'] = 'eve'
    return return_template(session, prefill=True)


@app.route('/blank_eve', methods=['POST'])
def blank_eve():
    session['object_type'] = 'eve'
    return return_template(session, prefill=False)


@app.route('/filled_npc', methods=['POST'])
def filled_npc():
    session['object_type'] = 'npc'
    return return_template(session, prefill=True)


@app.route('/blank_npc', methods=['POST'])
def blank_npc():
    session['object_type'] = 'npc'
    return return_template(session, prefill=False)


@app.route('/filled_geh', methods=['POST'])
def filled_geh():
    session['object_type'] = 'geh'
    return return_template(session, prefill=True)


@app.route('/blank_geh', methods=['POST'])
def blank_geh():
    session['object_type'] = 'geh'
    return return_template(session, prefill=False)


@app.route('/filled_gru', methods=['POST'])
def filled_gru():
    session['object_type'] = 'gru'
    return return_template(session, prefill=True)


@app.route('/blank_gru', methods=['POST'])
def blank_gru():
    session['object_type'] = 'gru'
    return return_template(session, prefill=False)


@app.route('/filled_bea', methods=['POST'])
def filled_bea():
    session['object_type'] = 'bea'
    return return_template(session, prefill=True)


@app.route('/blank_bea', methods=['POST'])
def blank_bea():
    session['object_type'] = 'bea'
    return return_template(session, prefill=False)


@app.route('/filled_geg', methods=['POST'])
def filled_geg():
    session['object_type'] = 'geg'
    return return_template(session, prefill=True)


@app.route('/blank_geg', methods=['POST'])
def blank_geg():
    session['object_type'] = 'geg'
    return return_template(session, prefill=False)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=80)
    app.run(debug=True)
