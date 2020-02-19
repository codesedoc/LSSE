import corpus.mrpc.mrpc as mrpc
from framework.base import FrameworkManager
# mrpc.test()

import utils.data_tool as data_tool
def data_loader_test():
    mrpc_obj = mrpc.get_mrpc_obj()
    data_manager = data_tool.DataManager(mrpc_obj)
    train_loader_tuple_list = data_manager.list_of_train_loader_tuple(k_fold=5, batch_size=2)

    test_loader = data_manager.test_loader(batch_size=8)

    for i, loader_tuple in enumerate(train_loader_tuple_list, 1):
        train_loader, valid_loader = loader_tuple
        print('loader_tuple{}: train_loader:{}  valid_loader:{}'.format(i, len(train_loader), len(valid_loader)))
        train_labels = [example.label for example in train_loader.dataset.examples]
        valid_labels = [example.label for example in valid_loader.dataset.examples]
        print('valid_labels: {} '.format(data_manager.calculate_distribution_of_two_class(valid_labels)))
        print('train_labels: {} '.format(data_manager.calculate_distribution_of_two_class(train_labels)))

    print('test_loader:{}'.format(len(test_loader)))
    test_labels = [example.label for example in test_loader.dataset.examples]
    print('test_labels: {} '.format(data_manager.calculate_distribution_of_two_class(test_labels)))


def framework_test():
    arg_dict = {
        'batch_size': 2,
        'learn_rate': 8e-6,
        # 'sgd_momentum': 0.4,
        'optimizer': 'adam',
        'k_fold': 5,
        'epoch': 5,
        'ues_gpu': 0,
        'repeat_train': True,
        'corpus': mrpc.get_mrpc_obj,
        'framework_name': "LSSE",
        'regular_flag': True,
    }

    framework_manager = FrameworkManager(arg_dict)
    framework_manager.train_model()


if __name__ == '__main__':
    framework_test()
    pass