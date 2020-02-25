import torch


class BertBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer_cased = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.model_cased = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.config = self.model_cased.config

    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch):
        last_hidden_states, pooled_output = self.model_cased(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch)
        return last_hidden_states, pooled_output


class BertForSeqClassify(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_cased = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-cased')
        self.tokenizer_cased = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch, labels):
        loss, outputs = self.model_cased(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch, labels=labels)
        return loss, outputs


class ALBertForSeqClassify(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_cased = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'albert-base-v2')
        self.tokenizer_cased = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.tokenizer_cased.do_lower_case = False

    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch, labels):
        loss, outputs = self.model_cased(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch, labels=labels)
        return loss, outputs

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
