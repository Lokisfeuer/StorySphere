from adventure import UnitId


def ask_chatgpt(old_adventure, newunit, id_to_name):
    adventure_text = old_adventure.to_text(id_to_name)
    unit_text = newunit.to_text(id_to_name)
    text = f'This is a story categorized in different units of different types:\n' \
           f'{adventure_text}\n' \
           f'I want this story to be a balanced '
