import torch
import torch.nn as nn



class RD_loss(nn.Module):

    def __init__(self):
        super(RD_loss, self).__init__()


    def compute_left(self,tensor):
        """
        get the 0 to n-2 cols of tensor(n cols totally)
        :param tensor: B*1*H*W size
        :return: the 0 to n-2 cols of tensor,B*1*H*(W-1)
        """
        left=tensor[:,:,:,:-1]
        return left

    def compute_right(self,tensor):
        """
        get the 1 to n-1 cols of tensor(n cols totally)
        :param tensor: B*1*H*W size
        :return: the 1 to n-1 cols of tensor,B*1*H*(W-1)
        """
        right=tensor[:,:,:,1:]
        return right

    def compute_top(self,tensor):
        """
        get the 0 to n-2 rows of tensor(n rows totally)
        :param tensor: B*1*H*W size
        :return: the 0 to n-2 rows of tensor,B*1*(H-1)*W
        """
        top=tensor[:,:,:-1,:]
        return top

    def compute_bottom(self,tensor):
        """
        get the 1 to n-1 rows of tensor(n rows totally)
        :param tensor: B*1*H*W size
        :return: the 1 to n-1 rows of tensor,B*1*(H-1)*W
        """
        bottom=tensor[:,:,1:,:]
        return bottom

    def compute_left_top(self,tensor):
        """
        get the [0 to n-2]*[0 to n-2] of tensor
        :param tensor: B*1*H*W size
        :return: the [0 to n-2]*[0 to n-2] of tensor,B*1*(H-1)*(W-1)
        """
        left_top=tensor[:,:,:,:-1][:,:,:-1,:]
        return left_top

    def compute_right_top(self,tensor):
        """
        get the [0 to n-2]*[1 to n-1] of tensor
        :param tensor: B*1*H*W size
        :return: the [0 to n-2]*[1 to n-1] of tensor,B*1*(H-1)*(W-1)
        """
        right_top=tensor[:,:,:,1:][:,:,:-1,:]
        return right_top

    def compute_bottom_left(self,tensor):
        """
        get the [1 to n-1]*[0 to n-2] of tensor
        :param tensor: B*1*H*W size
        :return: [1 to n-1]*[0 to n-2] of tensor,B*1*(H-1)*(W-1)
        """
        bottom_left=tensor[:,:,1:,:][:,:,:,:-1]
        return bottom_left

    def compute_bottom_right(self,tensor):
        """
        get the [1 to n-1]*[1 to n-1] of tensor
        :param tensor: B*1*H*W size
        :return: [1 to n-1]*[1 to n-1] of tensor,B*1*(H-1)*(W-1)
        """
        bottom_right=tensor[:,:,1:,:][:,:,:,1:]
        return bottom_right

    def compute_rd_top(self,b_tensor,t_tensor):
        """
        compute the relative depth between position[x,y] and position[x,y-1]
        :param b_tensor:bottom tensor,B*1*(H-1)*W
        :param t_tensor: top tensor,B*1*(H-1)*W
        :return: the relative depth map between position[x,y] and position[x,y-1]
        """
        depth_res_map=b_tensor-t_tensor
        added_depth_map=b_tensor+t_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map

    def compute_rd_bottom(self,t_tensor,b_tensor):
        """
        compute the relative depth between position[x,y] and position[x,y+1]
        :param t_tensor: top tensor,B*1*(H-1)*W
        :param b_tensor: bottom tensor,B*1*(H-1)*W
        :return: the relative depth map between position[x,y] and position[x,y+1]
        """
        depth_res_map=t_tensor-b_tensor
        added_depth_map=t_tensor+b_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map

    def compute_rd_left(self,l_tensor,r_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y]
        :param l_tensor: left tensor,B*1*H*(W-1)
        :param r_tensor: right tensor,B*1*H*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y]
        """
        depth_res_map=r_tensor-l_tensor
        added_depth_map=r_tensor+l_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map

    def compute_rd_right(self,l_tensor,r_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y]
        :param l_tensor: left tensor,B*1*H*(W-1)
        :param r_tensor: right tensor,B*1*H*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y]
        """
        depth_res_map=l_tensor-r_tensor
        added_depth_map=l_tensor+r_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map

    def compute_rd_left_top(self,l_t_tensor,b_r_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y-1]
        :param l_t_tensor: left top tensor,B*1*(H-1)*(W-1)
        :param b_r_tensor: bottom right tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y-1]
        """
        depth_res_map=b_r_tensor-l_t_tensor
        added_depth_map=b_r_tensor+l_t_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map

    def compute_rd_right_top(self,r_t_tensor,b_l_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y-1]
        :param r_t_tensor: right top tensor,B*1*(H-1)*(W-1)
        :param b_l_tensor: bottom left tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y-1]
        """
        depth_res_map=b_l_tensor-r_t_tensor
        added_depth_map=r_t_tensor+b_l_tensor
        scaled_relative_depth=depth_res_map/added_depth_map

        return scaled_relative_depth

    def compute_rd_bottom_left(self,b_l_tensor,r_t_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y+1]
        :param b_l_tensor: bottom left tensor,B*1*(H-1)*(W-1)
        :param r_t_tensor: right top tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y+1]
        """
        depth_res_map=r_t_tensor-b_l_tensor
        added_depth_map=r_t_tensor+b_l_tensor
        scaled_relative_depth=depth_res_map/added_depth_map

        return scaled_relative_depth

    def compute_rd_bottom_right(self,b_r_tensor,l_t_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y+1]
        :param b_r_tensor: bottom right tensor,B*1*(H-1)*(W-1)
        :param l_t_tensor: left top tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y+1]
        """
        depth_res_map=l_t_tensor-b_r_tensor
        added_depth_map=l_t_tensor+b_r_tensor
        scaled_relative_depth=depth_res_map/added_depth_map

        return scaled_relative_depth

    def compute_rd_map_list(self,depth_tensor):
        """
        compute all the relative depth map of depth_tensor
        :param depth_tensor: the original depth map,B*1*H*W
        :return: a list containing all the relative depth map of depth tensor,in the order of [top,right,bottom,left,left top,right top,bottom right,bottom left]
        """
        top=self.compute_top(depth_tensor)
        right=self.compute_right(depth_tensor)
        bottom=self.compute_bottom(depth_tensor)
        left=self.compute_left(depth_tensor)
        left_top=self.compute_left_top(depth_tensor)
        right_top=self.compute_right_top(depth_tensor)
        bottom_right=self.compute_bottom_right(depth_tensor)
        bottom_left=self.compute_bottom_left(depth_tensor)


        rd_top=self.compute_rd_top(bottom,top)
        rd_right=self.compute_rd_right(left,right)
        rd_bottom=self.compute_rd_bottom(top,bottom)
        rd_left=self.compute_rd_left(left,right)
        rd_left_top=self.compute_rd_left_top(left_top,bottom_right)
        rd_right_top=self.compute_rd_right_top(right_top,bottom_left)
        rd_bottom_right=self.compute_rd_bottom_right(bottom_right,left_top)
        rd_bottom_left=self.compute_rd_bottom_left(bottom_left,right_top)


        relative_depth_list=[rd_top,rd_right,rd_bottom,rd_left,rd_left_top,rd_right_top,rd_bottom_right,rd_bottom_left]

        return relative_depth_list

    def forward(self,pred,gt):
        """
        compute the difference of relative depth map between predicted depth map and ground truth depth map
        :param pred: predicted depth map,B*1*H*W
        :param gt: ground truth depth map,B*1*H*W
        :return: difference of relative depth map between pred and gt
        """

        pred_rd_list=self.compute_rd_map_list(pred)
        gt_rd_list=self.compute_rd_map_list(gt)

        loss_fn=torch.nn.L1Loss(reduction='mean')
        loss=0
        for i in range(len(pred_rd_list)):
            loss=loss+loss_fn(pred_rd_list[i],gt_rd_list[i])

        return loss





