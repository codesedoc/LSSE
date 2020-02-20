import utils.file_tool as file_tool
import corpus.base_corpus as base_corpus
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool


single_mrpc_obj = None

class Mrpc(base_corpus.Corpus):
    def create_examples(self):
        def create_examples_by_dicts(examples):
            example_obj_list = []
            example_obj_dict = {}
            for e in examples:
                sentence1_id = str(e['data'][0])
                sentence2_id = str(e['data'][1])
                sentence1 = self.sentence_dict[sentence1_id]
                sentence2 = self.sentence_dict[sentence2_id]

                id_ = str(e['id'])
                label = int(e['label'])
                example_obj = base_corpus.Example(id_, sentence1=sentence1, sentence2=sentence2, label=label)
                example_obj_list.append(example_obj)

                if id_ in example_obj_list:
                    raise ValueError("example in corpus is repeated")
                example_obj_dict[id_] = example_obj
            return example_obj_list, example_obj_dict

        train_examples = file_tool.load_data_pickle('corpus/mrpc/train_examples.pkl')
        self.train_example_list,self.train_example_dict  = create_examples_by_dicts(train_examples)

        test_examples = file_tool.load_data_pickle('corpus/mrpc/test_examples.pkl')
        self.test_example_list, self.test_example_dict = create_examples_by_dicts(test_examples)
        pass

    def create_sentences(self):
        original_sentence_dict = file_tool.load_data_pickle('corpus/mrpc/original_sentence_dict.pkl')
        self.sentence_list = []
        self.sentence_dict = {}
        for sent_id, o_sent in original_sentence_dict.items():
            sent_id = int(sent_id)
            sent_obj = base_corpus.Sentence(id_=sent_id, original_sentence=o_sent)
            self.sentence_list.append(sent_obj)
            if str(sent_id) in self.sentence_dict:
                raise ValueError("sentence in corpus is repeated")
            self.sentence_dict[str(sent_id)] = sent_obj
        pass

    def parse_sentences(self):
        parsed_sentence_dict = file_tool.load_data_pickle('corpus/mrpc/parsed_sentence_dict.pkl')
        if len(parsed_sentence_dict) != len(self.sentence_dict):
            raise ValueError("parsed_sentence_dict not march sentence_dict")

        if not general_tool.compare_two_dict_keys(self.sentence_dict.copy(), parsed_sentence_dict.copy()):
            raise ValueError("parsed_sentence_dict not march sentence_dict")

        for sent_id, parse_info in parsed_sentence_dict.items():
            sent_id = str(sent_id)
            self.sentence_dict[sent_id].parse_info = parse_info

        self.parse_info = parser_tool.process_parsing_sentence_dict(parsed_sentence_dict, modify_dep_name=True)
        numeral_sentence_dict = self.parse_info.numeral_sentence_dict

        if not general_tool.compare_two_dict_keys(self.sentence_dict.copy(), numeral_sentence_dict.copy()):
            raise ValueError("numeral_sentence_dict not march sentence_dict")

        for sent_id in self.sentence_dict.keys():
            self.sentence_dict[sent_id].syntax_info = numeral_sentence_dict[sent_id]

        print('the count of dep type:{}'.format(self.parse_info.dependency_count))
        print('the max len of sentence_tokens:{}'.format(self.parse_info.max_sent_len))

        pass


def get_mrpc_obj(force=False):
    global single_mrpc_obj
    if force or (single_mrpc_obj is None):
        single_mrpc_obj = Mrpc()
    return single_mrpc_obj


def test():
    mrpc = get_mrpc_obj()
    mrpc = get_mrpc_obj()
    mrpc = get_mrpc_obj()
