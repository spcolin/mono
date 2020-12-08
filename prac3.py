import torch


a=[[0,1,2,3,4,5,6,7,8,9]]*10
a=torch.tensor(a)
a=a.permute(1,0)
# print(a)

# 0 to n-1 row
top=a[:-1,:]

# 1 to n row
bottom=a[1:,:]

# 0 to n-1 col
left=a[:,:-1]

# 1 to n col
right=a[:,1:]


# [0 to n-2]*[0 to n-2]
left_top=a[:,:-1][:-1,:]

# [0 to n-2]*[1 to n-1]
right_top=a[:,1:][:-1,:]

# [1 to n-1]*[0 to n-2]
bottom_left=a[1:,:][:,:-1]

# [1 to n-1]*[1 to n-1]
bottom_right=a[1:,:][:,1:]
