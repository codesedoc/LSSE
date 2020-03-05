import utils.file_tool as file_tool
import utils.general_tool as general_tool


def extra_parsed_info_dict(filename):
    result = {}
    rows = file_tool.load_data(filename, 'r')
    for row in rows:
        items = row.split(' [Sq] ')
        if not general_tool.is_number(items[0]):
            raise ValueError
        result[items[0]] = row
    return result


def merge_special_and_old_file(special_file, old_file, newfile):
    old_parsed_info = extra_parsed_info_dict(old_file)
    special_parsed_info = extra_parsed_info_dict(special_file)
    old_parsed_info.update(special_parsed_info)

    save_data = []
    for sent_id, info in old_parsed_info.items():
        save_data.append(info)

    file_tool.save_list_data(save_data, newfile, 'w', need_new_line=False)


if __name__ == '__main__':
    merge_special_and_old_file('parsed_sentences(special).txt', 'parsed_sentences(old).txt', 'parsed_sentences(new).txt')