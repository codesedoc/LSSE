import utils.file_tool as file_tool
import matplotlib.pyplot as plt
import numpy as np


def run():
    # org_loss_list = file_tool.load_data_pickle('/home/sheng/Documents/study/workspace/python/PIRs/analysis/baseline/run on rtx/cuda/loss_list.pkl')
    # my_loss_list = file_tool.load_data_pickle('/home/sheng/Documents/study/workspace/python/PIRs/analysis/baseline/run on rtx/cuda/loss_list_My.pkl')

    org_loss_list = file_tool.load_data_pickle(
        '/home/sheng/Documents/study/workspace/python/PIRs/analysis/baseline/run_on_my_pc/cuda/loss_list.pkl')
    my_loss_list = file_tool.load_data_pickle(
        '/home/sheng/Documents/study/workspace/python/PIRs/analysis/baseline/run_on_my_pc/cuda/loss_list_my.pkl')

    pass
    fig = plt.figure()
    ax = plt.subplot()
    x = np.arange(len(org_loss_list))

    org_loss_np = np.array(org_loss_list)
    my_loss_np = np.array(my_loss_list)
    sub_loss_np = org_loss_np - my_loss_np
    # ax.scatter(x, org_loss_np, c='green', alpha=0.6)
    # ax.scatter(x, my_loss_np, c='red', marker='v', alpha=0.7)
    ax.scatter(x, sub_loss_np, c='red', marker='v', alpha=0.7)
    print("var:{}\t mean:{}\t ptp:{}".format(np.var(sub_loss_np), np.mean(sub_loss_np), np.ptp(sub_loss_np)))
    plt.show()

    pass

if __name__ == '__main__':
    run()