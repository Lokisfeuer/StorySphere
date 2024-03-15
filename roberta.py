from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


TOKENIZER = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
MODEL = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')


# returns a high-dimensional (1024) vector representation of the passed in sentence.
def roberta(sentence):
    # from https://huggingface.co/sentence-transformers/all-roberta-large-v1

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    sentences = [sentence]

    # Load model from HuggingFace Hub
    # I made this global variables because they take years to load so best just do it once.

    # Tokenize sentences
    encoded_input = TOKENIZER(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = MODEL(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # print("Sentence embeddings:")
    # print(sentence_embeddings)
    return sentence_embeddings.tolist()[0]
