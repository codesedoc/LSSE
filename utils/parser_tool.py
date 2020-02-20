import utils.number_tool as number_tool
import numpy as np


class ParseInfo:
    def __init__(self, **kwargs):
        super().__init__()

        self.numeral_sentence_dict = kwargs['numeral_sentence_dict']
        self.dependency_dict = kwargs['dependency_dict']
        self.dependency_count = len(self.dependency_dict)
        self.id2dependency = kwargs['id2dependency']
        max_sent_len = 0
        for number_sentence in self.numeral_sentence_dict.values():
            tokens = number_sentence['words']
            if len(tokens)>max_sent_len:
                max_sent_len = len(tokens)
        self.max_sent_len = max_sent_len


def modify_dependency_name(parsing_sentence_dict):
    for sent_id, parsing_info in parsing_sentence_dict.items():
        for dependency in parsing_info['dependencies']:
            name = dependency['name']
            name = name.split(':')
            if len(name) > 1:
                dependency['name'] = name[0]
    return parsing_sentence_dict


def process_parsing_sentence_dict(parsing_sentence_dict, modify_dep_name=False):
    if modify_dep_name:
        parsing_sentence_dict = modify_dependency_name(parsing_sentence_dict)
    word_dict = {}
    dependency_dict = {}

    error_dependencies_dict = {}
    count_of_dependencies = 0
    for sent_id, parsing_info in parsing_sentence_dict.items():
        for word in parsing_info['words']:
            if word in word_dict:
                word_info = word_dict[word]
                word_info['frequency'] += 1
            else:
                word_info = {'word': word, 'frequency': 1}
                word_dict[word] = word_info

        error_dependencies_list = []
        for dependency in parsing_info['dependencies']:
            count_of_dependencies += 1
            first_index = dependency['word_pair']['first']['index']
            second_index = dependency['word_pair']['second']['index']
            if (not number_tool.is_number(first_index)) or (not number_tool.is_number(second_index)):
                error_dependencies_list.append(dependency)
                continue
            if dependency['name'] in dependency_dict:
                dependency_info = dependency_dict[dependency['name']]
                dependency_info['frequency'] += 1
            else:
                dependency_info = {'dependency': dependency, 'frequency': 1}
                dependency_dict[dependency['name']] = dependency_info

        if len(error_dependencies_list) > 0:
            error_dependencies_dict[sent_id] = error_dependencies_list

    count_error_dependencies = 0
    for error_dependencies in error_dependencies_dict.values():
        count_error_dependencies += len(error_dependencies)

    count_temp = 0
    for dep in dependency_dict.values():
        count_temp += dep['frequency']
    if count_temp + count_error_dependencies != count_of_dependencies:
        raise ValueError
    print("The count of sentence: {}".format(len(parsing_sentence_dict)))
    print("The count of sentence including error_dependency: {}\terror_rate:{}".format(len(error_dependencies_dict),
                                                                                       round(len(
                                                                                           error_dependencies_dict) / len(
                                                                                           parsing_sentence_dict), 4)))
    print("The count of dependencies: {}\terror_rate:{}".format(count_of_dependencies,
                                                                round(count_error_dependencies / count_of_dependencies,
                                                                      4)))
    print("The count of error_dependencies: {}".format(count_error_dependencies))

    word_id_base = 0
    dependency_id_base = 0
    word_count = len(word_dict)
    dependency_count = len(dependency_dict)
    id2word_dict = {}
    id2dependency_dict = {}

    for i, word in enumerate(word_dict.keys(), word_id_base):
        word_info = word_dict[word]
        if str(i) in id2word_dict:
            raise ValueError
        word_info['id'] = i
        id2word_dict[str(i)] = word_info

    for i, dependency in enumerate(dependency_dict, dependency_id_base):
        dependency_info = dependency_dict[dependency]
        if str(i) in id2dependency_dict:
            raise ValueError
        dependency_info['id'] = i
        id2dependency_dict[str(i)] = dependency_info

    if len(id2word_dict) != word_count or len(id2dependency_dict) != dependency_count:
        raise RuntimeError

    numeral_sentence_dict = {}
    error_parsing_sentence = []
    for sent_id, parsing_info in parsing_sentence_dict.items():
        words = parsing_info['words']
        dependencies = parsing_info['dependencies']
        sentence_len = len(words)

        numeral_words = []
        numeral_dependencies = []

        for word in words:
            numeral_words.append(int(word_dict[word]['id']))

        max_index = 0
        for dep_info in dependencies:
            dep_id = str(dependency_dict[dep_info['name']]['id'])
            word_pair = dep_info['word_pair']
            first_word = word_pair['first']
            second_word = word_pair['second']
            first_index = first_word['index']
            second_index = second_word['index']
            if (not number_tool.is_number(first_index)) or (not number_tool.is_number(second_index)):
                continue
            numeral_dependencies.append((int(first_index), int(second_index), int(dep_id)))

            if int(first_index) > max_index:
                max_index = int(first_index)
            if int(second_index) > max_index:
                max_index = int(second_index)
        if max_index > sentence_len - 1:
            error_parsing_sentence.append(parsing_info)

        numeral_sentence_dict[sent_id] = {
            'words': numeral_words,
            "dependencies": numeral_dependencies
        }
        # numeral_dict

    error_parsing_sentence_ids = [str(x["id"]) for x in error_parsing_sentence]
    if len(error_parsing_sentence_ids) > 0:
        # file_tool.save_list_data(error_parsing_sentence_ids,
        #                          file_abs_path + '/proceess_files/error_parsing_sentence_ids.txt', 'w')
        raise ValueError

    return ParseInfo(numeral_sentence_dict=numeral_sentence_dict, dependency_dict=dependency_dict,
                     id2dependency=id2dependency_dict)


def dependencies2adj_matrix(numeral_dependencies, max_dep_kind, max_sent_len):
    result = np.zeros((max_dep_kind, max_sent_len, max_sent_len), dtype=np.int)
    numeral_dependencies = np.array(numeral_dependencies, dtype=np.int)
    dependencies_temp = numeral_dependencies[:, (2, 0, 1)]
    for dependency in dependencies_temp:
        result[tuple(dependency.reshape(-1, 1).tolist())] = 1
    return result


