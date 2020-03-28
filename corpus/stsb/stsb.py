import utils.file_tool as file_tool
import corpus.base_corpus as base_corpus
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool
import matplotlib.pyplot as plt
import numpy as np


single_stsb_obj = None


class Stsb(base_corpus.Corpus):
    data_path = 'corpus/stsb'

    def __init__(self):
        self.dev_example_list = None
        self.dev_example_dict = None
        self.example_id_interval = 10000
        super().__init__()


    def __extra_examples_from_org_file__(self, org_file, des_filename):
        if file_tool.check_file(des_filename):
            examples_dicts = file_tool.load_data_pickle(des_filename)
            return examples_dicts
        example_dicts = []
        rows = file_tool.read_csv(org_file, delimiter='\t')
        for i, row in enumerate(rows):

            if len(row) != 7 and len(row) != 9:
                raise RuntimeError

            example_temp = {
                'sent1': str(row[5]).strip(),
                'sent2': str(row[6]).strip(),
                'label': float(row[4]),
                'id': i,
            }

            example_dicts.append(example_temp)

        file_tool.save_data_pickle(example_dicts, des_filename)
        return example_dicts

    def create_examples(self):
        def create_examples_by_dicts(examples, id_base=0):
            example_obj_list = []
            example_obj_dict = {}

            # if len(examples)*2 >= self.example_id_interval:
            #     raise ValueError
            for e in examples:
                sent1 = str(e['sent1'])
                sent2 = str(e['sent2'])

                sent_obj1 = sent_obj_dict[str(sentence_id_dict[sent1])]
                sent_obj2 = sent_obj_dict[str(sentence_id_dict[sent2])]

                id_ = str(id_base + int(e['id']))
                label = float(e['label'])
                example_obj = base_corpus.Example(id_, sentence1=sent_obj1, sentence2=sent_obj2, label=label)
                example_obj_list.append(example_obj)

                if id_ in example_obj_list:
                    raise ValueError("example in corpus is repeated")
                example_obj_dict[id_] = example_obj

            if len(example_obj_list) != len(example_obj_dict):
                raise ValueError("example in corpus is repeated")

            return example_obj_list, example_obj_dict

        dev_dicts = self.__extra_examples_from_org_file__(
                                file_tool.connect_path(self.data_path, 'sts-dev.csv'),
                                file_tool.connect_path(self.data_path, 'dev_dicts.pkl'))

        train_dicts = self.__extra_examples_from_org_file__(
                                file_tool.connect_path(self.data_path, 'sts-train.csv'),
                                file_tool.connect_path(self.data_path, 'train_dicts.pkl'))

        test_dicts = self.__extra_examples_from_org_file__(
                                file_tool.connect_path(self.data_path, 'sts-test.csv'),
                                file_tool.connect_path(self.data_path, 'test_dicts.pkl'))

        example_dicts = train_dicts.copy() + test_dicts.copy() + dev_dicts.copy()
        sentence_id_dict = {}
        s_id = 0
        for e in example_dicts:
            sent1 = str(e['sent1'])
            sent2 = str(e['sent2'])

            if sent1 not in sentence_id_dict:
                sentence_id_dict[sent1] = s_id
                s_id += 1

            if sent2 not in sentence_id_dict:
                sentence_id_dict[sent2] = s_id
                s_id += 1
        sent_obj_dict = {}
        for org_sent, sent_id in sentence_id_dict.items():
            sent_obj_dict[str(sent_id)] = base_corpus.Sentence(id_=str(sentence_id_dict[org_sent]),
                                                               original_sentence=org_sent)

        self.train_example_list, self.train_example_dict = create_examples_by_dicts(train_dicts)


        self.test_example_list, self.test_example_dict = create_examples_by_dicts(test_dicts,
                                                                                  id_base=self.example_id_interval)

        self.dev_example_list, self.dev_example_dict = create_examples_by_dicts(dev_dicts,
                                                                                id_base=self.example_id_interval * 2)
        pass

    def get_example_list(self):
        return self.train_example_list.copy() + self.test_example_list.copy() + self.dev_example_list.copy()

    def get_example_dict(self):
        result = {}
        result.update(self.train_example_dict.copy())
        result.update(self.test_example_dict.copy())
        result.update(self.dev_example_dict.copy())

        return result

    def create_sentences(self):
        example_list = self.get_example_list()
        self.sentence_list = []
        self.sentence_dict = {}
        original_sentence_set = set()
        sentence_set = set()
        for example in example_list:
            self.sentence_dict[example.sentence1.id] = example.sentence1
            self.sentence_dict[example.sentence2.id] = example.sentence2
            original_sentence_set.add(example.sentence1.original)
            original_sentence_set.add(example.sentence2.original)
            sentence_set.add(example.sentence1)
            sentence_set.add(example.sentence2)

        for sentence in self.sentence_dict.values():
            self.sentence_list.append(sentence)


        if len(self.sentence_list) != len(self.sentence_dict):
            raise ValueError

        if len(original_sentence_set) != len(self.sentence_dict):
            raise ValueError

        if len(original_sentence_set) != len(sentence_set):
            raise ValueError
        pass

    def parse_sentences(self):
        parsed_sentence_org_file = file_tool.connect_path(self.data_path, 'parsed_sentences.txt')
        parsed_sentence_dict_file = file_tool.connect_path(self.data_path, 'parsed_sentence_dict.pkl')
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

    def create_data(self):
        self.create_examples()
        self.create_sentences()
        self.parse_sentences()
        self.save_original_sentence()
        self.modify_data()
        self.data_type = "real_num"

    def save_original_sentence(self):
        save_data = []
        for sentence in self.sentence_list:
            save_data.append('{}\t{}'.format(str(sentence.id), sentence.original))

        file_tool.save_list_data(save_data, file_tool.connect_path(self.data_path, 'original_sentence.txt'), 'w')

    def sentence_dict_from_examples(self):
        sentence_dict = {}
        for e in self.get_example_list():
            sentence_dict[e.sentence1.id] = e.sentence1
            sentence_dict[e.sentence2.id] = e.sentence2

        return sentence_dict

    def show_pared_info(self):
        print('the count of dep type:{}'.format(self.parse_info.dependency_count))
        print('the max len of sentence_tokens:{}, correspond sent id:{}'.format(self.parse_info.max_sent_len,
                                                                                self.parse_info.max_sent_id))
        print('the average len of sentence_tokens:{}'.format(self.parse_info.avg_sent_len))
        sent_len_table = self.parse_info.sent_len_table
        file_tool.save_data_pickle(sent_len_table, file_tool.connect_path(self.data_path, "sent_len_table.pkl"))
        plt.bar(range(1, len(sent_len_table) + 1), sent_len_table)
        plt.title("sentence tokens length distribution")
        plt.show()

    def modify_data(self):
        while(True):
            print("Whether deleted examples with too long sentence, y/n?")
            delete_flag = input()
            if delete_flag == 'y':
                self.__delete_examples_by_sent_len_threshold__(50)
            elif delete_flag == 'n':
                break

    def __delete_examples_by_sent_len_threshold__(self, threshold):
        def delete_examples_from_dict(example_dict, example_ids):
            deleted_es = []
            for e_id in example_ids:
                deleted_es.append(example_dict.pop(str(e_id)))
            # print("deleted {} examples".format(len(deleted_es)))
            return example_dict

        def delete_examples(name, example_list, example_dict):
            old_count = len(example_dict)

            delete_examples = self.__collect_examples_over_threshold__(example_list, threshold)

            delete_examples_from_dict(example_dict, [e.id for e in delete_examples])

            example_list = list(example_dict.values())

            count = len(example_dict)

            if count + len(delete_examples) != old_count:
                raise ValueError("deleted train data error")

            print('deleted {} {} examples'.format(len(delete_examples), name))

            return example_list, example_dict

        self.sent_distribute_count()

        print("length threshold is {}".format(threshold))

        # train_old_count = len(self.train_example_dict)
        # test_old_count = len(self.test_example_dict)
        #
        # train_delete_examples = self.__collect_examples_over_threshold__(self.train_example_list, threshold)
        # test_delete_examples = self.__collect_examples_over_threshold__(self.test_example_list, threshold)
        #
        # delete_examples_from_dict(self.train_example_dict, [e.id for e in train_delete_examples])
        # delete_examples_from_dict(self.test_example_dict, [e.id for e in test_delete_examples])
        #
        # self.train_example_list = list(self.train_example_dict.values())
        # self.test_example_list = list(self.test_example_dict.values())
        #
        # train_count = len(self.train_example_dict)
        # test_count = len(self.test_example_dict)
        #
        # if train_count + len(train_delete_examples) != train_old_count:
        #     raise ValueError("deleted train data error")
        #
        # if test_count + len(test_delete_examples) != test_old_count:
        #     raise ValueError("deleted test data error")
        #
        # for e_id in self.test_example_dict.keys():
        #     if e_id in self.train_example_dict:
        #         raise ValueError("example {} in both test and train".format(e_id))
        #
        # print('deleted {} train examples'.format(len(train_delete_examples)))
        # print('deleted {} test examples'.format(len(test_delete_examples)))

        self.train_example_list, self.train_example_dict = delete_examples('train', self.train_example_list, self.train_example_dict)
        self.test_example_list, self.test_example_dict = delete_examples('test', self.test_example_list, self.test_example_dict)
        self.dev_example_list, self.dev_example_dict = delete_examples('dev', self.dev_example_list, self.dev_example_dict)

        self.sent_distribute_count()
        self.max_sent_len = threshold

    def get_max_sent_len(self):
        return self.max_sent_len

    def __collect_examples_over_threshold__(self, examples, threshold):
        result_examples = []
        for e in examples:
            if (e.sentence1.len_of_tokens() > threshold) or (e.sentence2.len_of_tokens() > threshold):
                result_examples.append(e)
        return result_examples

    def sent_distribute_count(self):
        def data_set_count(name, example_list):
            count = len(example_list)
            out_count = len(self.__collect_examples_over_threshold__(example_list, length_threshold))
            print('{} data: {}/{}, rate:{}'.format(name, count, out_count,
                                                   round((count - out_count) / count, 6)))

        sent_len_table = np.zeros(500, dtype=np.int)
        for s in self.sentence_list:
            sent_len_table[s.len_of_tokens()] += 1

        sent_count = sent_len_table.sum()

        train_count = len(self.train_example_list)
        test_count = len(self.test_example_list)
        dev_count = len(self.dev_example_list)

        print('count of sentence:{}'.format(sent_count))
        print('count of example:{}'.format(train_count + test_count + dev_count))
        while(True):
            print('please input length threshold, "e" donate exit')
            length_threshold = input()
            if length_threshold == "e":
                break
            if not general_tool.is_number(length_threshold):
                continue

            length_threshold = int(length_threshold)
            sent_temp = sent_len_table[:length_threshold + 1].sum()
            print('sentence: {}/{}, rate:{}'.format(sent_temp, sent_count-sent_temp, round(sent_temp/sent_count, 6)))

            # train_out_count = len(self.__collect_examples_over_threshold__(self.train_example_list, length_threshold))
            #
            # test_out_count = len(self.__collect_examples_over_threshold__(self.test_example_list, length_threshold))
            #
            # print('train data: {}/{}, rate:{}'.format(train_count, train_out_count,
            #                                           round((train_count-train_out_count)/train_count, 6)))
            #
            # print('test data: {}/{}, rate:{}'.format(test_count, test_out_count,
            #                                          round((test_count-test_out_count)/test_count, 6)))

            data_set_count('train', self.train_example_list)
            data_set_count('test', self.test_example_list)
            data_set_count('dev', self.dev_example_list)


def get_stsb_obj(force=False):
    global single_stsb_obj
    single_stsb_obj_file = file_tool.connect_path("corpus/stsb", 'stsb_obj.pkl')

    if file_tool.check_file(single_stsb_obj_file):
        single_stsb_obj = file_tool.load_data_pickle(single_stsb_obj_file)

    if force or (single_stsb_obj is None):
        single_stsb_obj = Stsb()
        file_tool.save_data_pickle(single_stsb_obj, single_stsb_obj_file)

    print('the count of dep type:{}'.format(single_stsb_obj.parse_info.dependency_count))
    print('the max len of sentence_tokens:{}'.format(single_stsb_obj.get_max_sent_len()))

    return single_stsb_obj


def test():
    stsb = get_stsb_obj()
    # stsb.show_pared_info()

