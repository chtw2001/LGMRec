# import datetime
# cur = datetime.datetime.now()
# cur = cur.strftime('%b-%d-%Y-%H-%M-%S')
# print(cur)
import torch

list_ = []
list_.extend(3*[1])
list_.extend(2*[2])
items = [1,2,45,5]
items_2 = [1,3,3,3,1]
list_.extend(items)
list_.extend(items_2)
list_2 = list_.copy()
print(torch.Tensor([list_, list_2]))