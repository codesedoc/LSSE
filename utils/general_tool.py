import math
import numpy as np


def compare_two_dict_keys(dict1, dict2):
    for key in dict1.keys():
        if key not in dict2:
            return False

    for key in dict2.keys():
        if key not in dict1:
            return False
    return True


global_position_encodings = None


def get_position_encodings(length, dimension):
    position_encodings = []
    for t in range(length):
        position_encoding = []
        for d in range(dimension):
            if d % 2 == 0:
                d = int(d / 2)
                wd = 1/(math.pow(10000, 2*d/dimension))
                position_encoding.append(math.sin(wd * t))
                position_encoding.append(math.cos(wd * t))
        position_encodings.append(position_encoding)
    position_encodings = np.array(position_encodings)
    return position_encodings


def get_global_position_encodings(length=100, dimension=300):
    global global_position_encodings
    if global_position_encodings is None:
        global_position_encodings = get_position_encodings(length, dimension)
    return global_position_encodings


# def get_sinusoid_encoding_table(n_position, d_hid):
#     ''' Sinusoid position encoding table '''
#
#     def cal_angle(position, hid_idx):
#         return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
#
#     def get_posi_angle_vec(position):
#         return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
#
#     sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
#
#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  偶数正弦
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  奇数余弦
#     return sinusoid_table

def test():
    result1 = get_global_position_encodings(10, 6)
    # result2 = get_sinusoid_encoding_table(10, 6)
    pass

if __name__ == '__main__':
    test()

