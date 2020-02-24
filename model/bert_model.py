import torch


class BertBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.tokenizer_cased = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch):
        last_hidden_states, pooler_output = self.bert( input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch)

        if torch.isnan(last_hidden_states).sum() > 0:
            print(torch.isnan(last_hidden_states))
            raise ValueError

        cls_state = last_hidden_states[:, 0]
        sentence1_states = last_hidden_states[:, 1:1]
        sentence2_states = last_hidden_states[:, 1 + 1: -1]
        return cls_state, sentence1_states, sentence2_states, pooler_output


class ALBertBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'albert-base-v2')
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'albert-base-v2')
        # default lower_case
        # self.tokenizer.do_lower_case = False
        pass

    def forward(self, sentence_tokens, segment_ids, sep_index):
        last_hidden_states, _ = self.model(sentence_tokens, token_type_ids=segment_ids)

        if torch.isnan(last_hidden_states).sum() > 0:
            print(torch.isnan(last_hidden_states))
            raise ValueError

        cls_state = last_hidden_states[:, 0]
        sentence1_states = last_hidden_states[:, 1:sep_index]
        sentence2_states = last_hidden_states[:, sep_index + 1: -1]
        return cls_state, sentence1_states, sentence2_states
