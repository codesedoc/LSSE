import utils.file_tool as file_tool
import corpus.base_corpus as base_corpus
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool


single_mrpc_obj = None


class Mrpc(base_corpus.Corpus):
    data_path = 'corpus/mrpc'

    def __extra_examples_from_org_file__(self, org_file, des_filename):
        if file_tool.check_file(des_filename):
            examples_dicts = file_tool.load_data_pickle(des_filename)
            return examples_dicts
        example_dicts = []
        rows = file_tool.load_data(org_file, mode='r')
        examples_id = 0
        for i, row in enumerate(rows):
            result = row.split("\t")
            if i == 0:
                continue
            if len(result) != 5:
                raise RuntimeError
            example_temp = {
                'sent_id1': int(result[1]),
                'sent_id2': int(result[2]),
                'label': int(result[0]),
                'id': examples_id
            }
            example_dicts.append(example_temp)
            examples_id += 1
        file_tool.save_data_pickle(example_dicts, des_filename)
        return example_dicts

    def create_examples(self):
        def create_examples_by_dicts(examples):
            example_obj_list = []
            example_obj_dict = {}
            for e in examples:
                sentence1_id = str(e['sent_id1'])
                sentence2_id = str(e['sent_id2'])
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

        train_dicts = self.__extra_examples_from_org_file__(
                                file_tool.connect_path(self.data_path, 'train.txt'),
                                file_tool.connect_path(self.data_path, 'train_dicts.pkl'))

        test_dicts = self.__extra_examples_from_org_file__(
                                file_tool.connect_path(self.data_path, 'test.txt'),
                                file_tool.connect_path(self.data_path, 'test_dicts.pkl'))

        self.train_example_list, self.train_example_dict = create_examples_by_dicts(train_dicts)

        self.test_example_list, self.test_example_dict = create_examples_by_dicts(test_dicts)

        pass

    def __extra_sentences_from_org_file__(self, org_file, des_filename):
        if file_tool.check_file(des_filename):
            sentence_dict = file_tool.load_data_pickle(des_filename)
            return sentence_dict
        sentence_dict = {}
        rows = file_tool.load_data(org_file, mode='r')
        for i, row in enumerate(rows):
            result = row.split("\t")
            if i == 0:
                continue
            if len(result) != 7:
                raise RuntimeError

            if not general_tool.is_number(result[0]):
                raise RuntimeError

            if str(result[0]) in sentence_dict:
                raise RuntimeError

            sentence_dict[str(result[0])] = str(result[1])

        file_tool.save_data_pickle(sentence_dict, des_filename)

        return sentence_dict

    def create_sentences(self):
        original_sentence_dict = self.__extra_sentences_from_org_file__(
                                file_tool.connect_path(self.data_path, 'data.txt'),
                                file_tool.connect_path(self.data_path, 'original_sentence_dict.pkl'))
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
        parsed_sentence_org_file = 'corpus/mrpc/parsed_sentences.txt'
        parsed_sentence_dict_file = 'corpus/mrpc/parsed_sentence_dict.pkl'
        if file_tool.check_file(parsed_sentence_dict_file):
            parsed_sentence_dict = file_tool.load_data_pickle(parsed_sentence_dict_file)
        else:
            parsed_sentence_dict = parser_tool.extra_parsed_sentence_dict_from_org_file(parsed_sentence_org_file)
            file_tool.save_data_pickle(parsed_sentence_dict, parsed_sentence_dict_file)

        if len(parsed_sentence_dict) != len(self.sentence_dict):
            # raise ValueError("parsed_sentence_dict not march sentence_dict")
            pass

        if not general_tool.compare_two_dict_keys(self.sentence_dict.copy(), parsed_sentence_dict.copy()):
            raise ValueError("parsed_sentence_dict not march sentence_dict")

        # for sent_id, info in parsed_sentence_dict.items():
        #     if info['original'] != self.sentence_dict[sent_id].original:
        #         raise ValueError("parsed_sentence_dict not march sentence_dict")

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

    def sentence_dict_from_examples(self):
        sentence_dict = {}
        for e in self.train_example_dict.values():
            sentence_dict[e.sentence1.id] = e.sentence1
            sentence_dict[e.sentence2.id] = e.sentence2

        for e in self.test_example_dict.values():
            sentence_dict[e.sentence1.id] = e.sentence1
            sentence_dict[e.sentence2.id] = e.sentence2

        return sentence_dict

def get_mrpc_obj(force=False):
    global single_mrpc_obj
    if force or (single_mrpc_obj is None):
        single_mrpc_obj = Mrpc()
    return single_mrpc_obj


def test():
    mrpc = get_mrpc_obj()
    mrpc = get_mrpc_obj()
    mrpc = get_mrpc_obj()
