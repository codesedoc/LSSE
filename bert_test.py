import torch



model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased', do_lower_case=False)

def token_sent(sentence1, sentence2):
    inputs_ls_cased = tokenizer.encode_plus(sentence1, sentence2)
    input_ids = inputs_ls_cased["input_ids"]
    print(input_ids)
    revised_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(revised_tokens)

# How do I make friends.	How to make friends?How do I choose a journal to publish my paper?	Where do I publish my paper?
token_sent('How do I get off Quora?', 'How do get out of Quora?')
