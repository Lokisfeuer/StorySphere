## Adventure in Computer-Readable Format
from Adventure-class object to a list of numbers
(= point in a high-dimensional vector space) and back.
### The Forward Process
- for each object in adventure generate pre-encoding.
Save these pres in dict id_to_pre
- for each object in adventure generate real encoding.
    - use pretrained-rnn Anna-encoder for list of ids. 
- for each object class (spi, npc, ...) use 
pretrained-rnn Bernd-encoder and to generate encoding for the
list of objects of a certain class.
- concat all these encodings. 
- use an autoencoder (Charlene-encoder) to reduce the amount 
of numbers (= dimensions)

*maybe use instead of Bernd mutliple different rnns. (Mr fragen)

### The Backward Process
- use Charlene-decoder
- slice it into the different class-encodings
- use bernd-decoder to get multiple objects real encoding
from class encodings (real-encoded).
- generate a key for each object by ignoring everything 
where pre-encodings of other objects would be so that we 
get something similar to their pre-encoding
- generate a dict with {key: id} 
(assign ids to real-encodings, then assing their ids to their keys)
- Compare everything from the real-encodings where pre-encoding 
would be with the different keys to find the right id. 
- Replace
pre-encodings with ids.
- Generate adventure-class object

## Reinforcement Learning
Storyprompting. Take some prompts (not only text) from the user
and feed them in as well. (Not part of action-space). 
At some point maybe add user-groups and train AI's on these.
- Environment: Adventure-encoding Vektor space
- State: encoded Adventure
- Actions: move adventure a little (?) within vector space.
- Reward: Value of adventure after action - before action. 
  - punish deleting or changing already existing objects heavily.
- Policy: Gets trained with the above.

### The Evaluation Function
The value of an adventure needs to be determined
through some evaluation function that takes the
adventure in any format and returns its value.
The quality of the policy's predictions is mostly 
limited by the quality of this evaluation function.
At the start I intend to use the handcrafted evaluation
function which I wrote (WHERE?). Later it should be 
replaced by human data.

#### Handcrafted Evaluation Function
TODO: add a more detailled explanation of the handcrafted evaluation function.
#### Human ranking
Generate a lot of different adventures that make sense automatically.
Use the old AI-RPG game to get humans ranking these adventures in a 
competitive system. Thereby assign adventures a ranking score and 
use it as their value. 
#### Final Function
Don't use an evaluation function at all but use direct rewards from
whether a user took the idea (good) or not (bad) to train the 
policy. This only works if a userbase already exists.


