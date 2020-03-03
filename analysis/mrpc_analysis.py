import corpus
import torch
import numpy as np


def test():
    mrpc_obj = corpus.mrpc.get_mrpc_obj()
    sentence_list = mrpc_obj.sentence_list

    bert_base_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    unk_token_rates = []
    for sentence in sentence_list:
        inputs_ls_cased = bert_base_tokenizer.encode_plus(sentence.word_tokens())
        input_ids = inputs_ls_cased["input_ids"]

        revised_tokens = bert_base_tokenizer.convert_ids_to_tokens(input_ids)

        unk_token_rate = 0
        for token in revised_tokens:
            if token == bert_base_tokenizer.unk_token:
                unk_token_rate += 1
        unk_token_rates.append(unk_token_rate/len(sentence.word_tokens()))
    unk_token_rates = np.array(unk_token_rates)
    print('avg_unk_token_rate:{}'.format(unk_token_rates.mean()))

    pass