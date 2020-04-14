import torch

def occupy_gpu():
    memory = []
    while(True):
        try:
            memory.append(torch.zeros((100,)*4, dtype=torch.double, device=torch.device('cuda', 0)))
        except Exception as e:
            # print(e)
            break
    return memory


if __name__ == '__main__':
    memory = occupy_gpu()
    print('finish apply memory')
    del memory[len(memory) - 1]
    while(True):
        memory[0] * memory[0]


    pass