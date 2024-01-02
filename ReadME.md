This project is not documented thoroughly (yet).

rnn.py contains a function called "train_model()"
which takes a list of data samples and returns an
encoder and a decoder. These can be saved with
torch.save() and loaded with torch.load().
These Decoders are Recurrent Neural Networks. 
Actually Transformers would probably be better
but RNNs should work fine.

encode_adventure.py contains 
- the adventure class with the central 
adventure-object structure. It is partially 
german and the structure is not in its final 
form yet.
  - A function demo_adventure() that writes an
  example adventure. This function is not up to
  date. Neither is the JSON file the function was
  adventure was written in.
- Functions to encode the objects of an adventure 
to either pre_encoding or the real encoding.
- Hopefully soon: A full automatic real encoding 
function that simply takes an adventure and returns 
the complete encoding of this adventure. This needs
a few more RNNs and data.

webapp.py sets up a website with a minimal user interface. 
This calls the get_adventure() function from main.py. This is
its interface to the rest of the program. Currently, that 
function is not up-to-date (or doesn't exist anymore) so it ain't
working. 
Webapp.py is purely backend - the frontend will be handled by 
another person from the team.

seq2seq.py is mostly copied from the tutorial to RNNs.
When run it creates an RNN encoder decoder pair and 
plots the training process. 