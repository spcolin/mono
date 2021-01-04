import torch


torch.distributed.init_process_group('nccl',
        init_method='env://')
local_rank=torch.distributed.get_rank()




print("successful!!!!")