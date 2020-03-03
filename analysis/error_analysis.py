import utils.file_tool as file_tool
import utils.general_tool as general_tool

error_file_path = 'analysis/error file'


class ErrorInfo:
    def __init__(self, name):
        self.name = name
        self.fn_example_dict = None
        self.fp_example_dict = None
        self.create_fn_fp_examples()

    def __extra_error_examples_from_org_file__(self, error_file):
        result = {}
        rows = file_tool.load_data(error_file, 'r')
        count = len(rows)//3
        for i in range(count):
            pair_index = i*3
            id_pair = rows[pair_index].strip()[1:-1].split(',')
            if len(id_pair) != 2:
                raise ValueError
            if not(general_tool.is_number(id_pair[0]) and general_tool.is_number(id_pair[0])):
                raise ValueError
            sentence1 = rows[pair_index+1].strip()
            sentence2 = rows[pair_index + 2].strip()
            result[str(id_pair)]={
                's1': sentence1,
                's2': sentence2
            }
        return result

    def create_fn_fp_examples(self):
        fn_file = file_tool.connect_path(error_file_path, self.name, 'fn_error_sentence_pairs.txt')
        fp_file = file_tool.connect_path(error_file_path, self.name, 'fp_error_sentence_pairs.txt')
        self.fn_example_dict= self.__extra_error_examples_from_org_file__(fn_file)
        self.fp_example_dict = self.__extra_error_examples_from_org_file__(fp_file)

class ErrorAnalyser:
    def __init__(self, our_error, others_error):
        self.our_error = our_error
        self.others_error = others_error

    def __get_U_sub_V__(self, dictU, dictV):
        result = {}
        for key in dictU.keys():
            if key not in dictV:
                result[key] = dictU[key]
        return result

    def __create_exampleU_sub_exampeV_file__(self, example_dictU , example_dictV, filename):
        save_example_dict = self.__get_U_sub_V__(example_dictU, example_dictV)
        save_data = ['total of examples: {}, {} examples not in the error examples of this method\n\n'.format(len(example_dictU), len(save_example_dict))]
        for key in save_example_dict.keys():
            save_data.append(key)
            save_data.append(save_example_dict[key]['s1'])
            save_data.append(save_example_dict[key]['s2'])
            if (key not in example_dictU) or (key in example_dictV):
                raise ValueError

        file_tool.save_list_data(save_data, filename, 'w')

    def create_difference_file_between_our_and_others(self):
        result_path = 'analysis/result'
        for other_error in self.others_error:
            difference_path = file_tool.connect_path(result_path, other_error.name)
            file_tool.makedir(difference_path)
            fn_result_file = file_tool.connect_path(difference_path, 'fn.txt')
            fp_result_file = file_tool.connect_path(difference_path, 'fp.txt')
            self.__create_exampleU_sub_exampeV_file__(self.our_error.fn_example_dict, other_error.fn_example_dict, fn_result_file)
            self.__create_exampleU_sub_exampeV_file__(self.our_error.fp_example_dict, other_error.fp_example_dict,
                                                      fp_result_file)


def test():
    our_error_info = ErrorInfo('LSSE')
    others_error_info = [
        ErrorInfo('LSSE-pe'),
        ErrorInfo('LSeE'),
        ErrorInfo('LSyE'),
        ErrorInfo('LSyE-pe'),
        ErrorInfo('SeE'),
        ErrorInfo('LE'),
    ]
    error_analyer = ErrorAnalyser(our_error_info, others_error_info)
    error_analyer.create_difference_file_between_our_and_others()