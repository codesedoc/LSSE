import torch


class BertBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        # self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.tokenizer_cased = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.bert_cased = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')


    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch):
        last_hidden_states, pooler_output = self.bert_cased(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch)

        if torch.isnan(last_hidden_states).sum() > 0:
            print(torch.isnan(last_hidden_states))
            raise ValueError

        cls_state = last_hidden_states[:, 0]
        sentence1_states = last_hidden_states[:, 1:1]
        sentence2_states = last_hidden_states[:, 1 + 1: -1]
        return cls_state, sentence1_states, sentence2_states, pooler_output


class BertForSeqClassify(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_cased = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-cased')
        self.tokenizer_cased = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch, labels):
        loss, outputs = self.model_cased(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch, labels=labels)

        # if torch.isnan(last_hidden_states).sum() > 0:
        #     print(torch.isnan(last_hidden_states))
        #     raise ValueError
        #
        # cls_state = last_hidden_states[:, 0]
        # sentence1_states = last_hidden_states[:, 1:1]
        # sentence2_states = last_hidden_states[:, 1 + 1: -1]
        return loss, outputs


class BertForSequenceClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bert_cased = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.tokenizer_cased = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.config.hidden_size, 2)
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()

        self.num_labels = 2
        # self.init_weights()


    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch, labels):

        outputs = self.bert_cased(
            input_ids_batch,
            attention_mask=attention_mask_batch,
            token_type_ids=token_type_ids_batch,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                # labels = 1 - labels
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                # loss = torch.nn.CrossEntropyLoss()(result, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


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
