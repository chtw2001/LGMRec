import datetime
import torch
import numpy as np
# cur = datetime.datetime.now()
# cur = cur.strftime('%b-%d-%Y-%H-%M-%S')
# print(cur)


# list_ = []
# list_.extend(3*[1])
# list_.extend(2*[2])
# items = [1,2,45,5]
# items_2 = [1,3,3,3,1]
# list_.extend(items)
# list_.extend(items_2)
# list_2 = list_.copy()
# print(torch.Tensor([list_, list_2]))

# decay = 0.0
# if decay is not None:
#     print(decay)

len_list = []
for i in range(10):
    len_list.append([i])
len_list = np.asarray(len_list)
# print(len_list.shape)

topk = []
for j in range(5):
    ex = []
    for i in range(5):
        ex.append(i)
    topk.append(ex)
print(topk)

topk = np.asarray(topk)
print(topk, topk.shape)

len_rank = np.full_like(len_list, topk.shape[1])
print(len_rank)
