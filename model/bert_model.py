import torch


class BertBase(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = torch.hub.load('huggingface/pytorch-transformers', 'config', self.args.model_name_or_path,
                                     num_labels=args.num_labels,
                                     finetuning_task=self.args.task_name,
                                     cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model',
                                    self.args.model_name_or_path,
                                    from_tf=bool(".ckpt" in self.args.model_name_or_path),
                                    config=self.config,
                                    cache_dir=self.args.cache_dir if self.args.cache_dir else None)
        self.name = self.config.model_type

    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch):
        last_hidden_states, pooled_output = self.model(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch)
        return last_hidden_states, pooled_output


class BertForSeqClassify(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = torch.hub.load('huggingface/pytorch-transformers', 'config', self.args.model_name_or_path,
                                num_labels=args.num_labels,
                                finetuning_task=self.args.task_name,
                                cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification',
                                        self.args.model_name_or_path,
                                        from_tf=bool(".ckpt" in self.args.model_name_or_path),
                                        config=self.config,
                                        cache_dir=self.args.cache_dir if self.args.cache_dir else None)
        self.name = self.config.model_type + '_for_classification'

    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch, labels):
        loss, outputs = self.model(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch, labels=labels)
        return loss, outputs


class ALBertForSeqClassify(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'albert-base-v2')
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'albert-base-v2')
        self.config = self.model.config
        self.name = self.config.model_type

    def forward(self, input_ids_batch, token_type_ids_batch, attention_mask_batch, labels):
        loss, outputs = self.model(input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch, labels=labels)
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
