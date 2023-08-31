# Design Idea
# Incoming a batch of sample rays and forward
import torch
import torch.nn as nn

from torch.autograd import Function


# import cuda_replacement as _backend
#
#
# # 假设一些参数
# # Number of Levels                   L              16
# # Max entries per level              T              2^14-2^24
# # number of feature per entry        F              2
# # Coarest resolution                 N_min          16
# # Finest resolution                  N_max          512 - 524288
# class _CudaAcceleration(Function):
#     def forward():
#         pass
#
#     def backward():
#         pass

class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        pass


class MyHash_replacement:
    def __init__(self, ):
        self.key = key
        self.value = value
        self.HASH_TYPE = HASH_TYPE

    def __hash__(self, ):
        pass

    def __eq__(self, other):
        pass

    # def grid_hash(self, pos_grid, HASH_TYPE):
    #     if HASH_TYPE == 'Prime':
    #         return self.prime_hash(pos_grid)
    #     elif HASH_TYPE == 'CoherentPrime':
    #         return self.coherent_prime_hash(pos_grid)
    #     elif HASH_TYPE == 'ReversePrime':
    #         return self.reverse_prime_hash(pos_grid)
    #     else:
    #         TypeError()
    #
    # def apply_hash(self, pos_grid, factors):
    #     assert len(pos_grid) <= len(factors), 'dimensions of pos_grid exceeding to the size of factors'
    #     result = 0
    #     for i in range(len(pos_grid)):
    #         result ^= pos_grid[i] * factors[i]
    #
    # def prime_hash(self, pos_grid):
    #     factors = list([1958374283, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737])
    #     result = self.apply_hash(pos_grid, factors)
    #     return result
    #
    # def coherent_prime_hash(self, pos_grid):
    #     factors = list([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737])
    #     result = self.apply_hash(pos_grid, factors)
    #     return result
    #
    # def reverse_prime_hash(self, pos_grid):
    #     factors = list([2165219737, 1434869437, 2097192037, 3674653429, 805459861, 2654435761, 1958374283])
    #     result = self.apply_hash(pos_grid, factors)
    #     return result


class HashGrid:
    def __init__(self, cfg_grid):
        # load,preset,initialize

        # 1/3 load
        self.L = cfg_grid['num_levels']  # 暂时无用，resolution是我预设好的
        self.T = cfg_grid['max_entries']
        self.F = cfg_grid['num_features']
        self.N_min = cfg_grid['min_resolution']
        self.N_max = cfg_grid['max_resolution']
        self.bound = torch.tensor(cfg_grid['bound'])

        # 2/3 preset
        self.grid_steps_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

        self.create_hash_grid()

        # 因为gpu的计算是分threads和blocks的。
        # 那么，分配以下；
        # Blocks是几个resolution。L个
        # Threads是几个entry。T个
        # 每个threads计算一个像素。

    def forward(self, batch):
        # 好，到现在为止，数据已经可以读取了，tracking不做的情况下，下面需要的就是最重要的mapping的部分，用explicit的方法来mapping
        # Now batch={dict:5}
        #   'frame_id'=Tensor(1)
        #   'c2w'=Tensor(4,4)
        #   'rgb'=Tensor(680*1200,3)
        #   'depth'=Tensor(680*1200)
        #   'direction'=Tensor(680*1200,3)
        # test
        # 这个batch用来让IDE识别batch里面到底是什么，有助于我写代码
        batch = {'frame_id': torch.zeros(1),
                 'c2w': torch.zeros((4, 4)),
                 'rgb': torch.zeros((680 * 1200, 3)),
                 'depth': torch.zeros((680 * 1200)),
                 'direction': torch.zeros((680 * 1200, 3))}

        # create output receiver
        feature_size = self.features_sample().size()
        grid_size = len(self.grid_steps_list)
        all_batch_features = torch.zeros((680 * 1200, grid_size, feature_size))
        # /test
        for i in range(batch['depth'].numel()):  # depth can get the number of items
            c2w = batch['c2w']
            ray_direction = batch['direction'][i]
            ray_depth = batch['depth'][i]
            point_location = self.get_ray_coordinate(c2w, ray_direction, ray_depth)

            for j in range(len(self.grid_steps_list)):
                x_idx_min = point_location[0] // self.grid_steps_list[j]
                y_idx_min = point_location[1] // self.grid_steps_list[j]
                z_idx_min = point_location[2] // self.grid_steps_list[j]
                x_idx_max = x_idx_min + 1
                y_idx_max = y_idx_min + 1
                z_idx_max = z_idx_min + 1

                this_features = self.trilinear_interpolation(x_idx_min, y_idx_min, z_idx_min,
                                                             x_idx_max, y_idx_max, z_idx_max,
                                                             point_location[0], point_location[1], point_location[2],
                                                             self.grid_list[j])
                all_batch_features[i, j, :] = this_features
            # now we have the features in each hash grid.
            # stored in all_batch_features with shape of [pixels,grid_size,feature_size]

            break

    def create_hash_grid(self):
        # 有grid_step就行了，用到的时候反正用的是hash来算的。在python中，用dict来指代就行。
        self.grid_list = []
        for i in range(len(self.grid_steps_list)):
            self.grid_list.append(dict())

        pass

    def trilinear_interpolation(self, x_idx_min, y_idx_min, z_idx_min,
                                x_idx_max, y_idx_max, z_idx_max,
                                x_val, y_val, z_val,
                                hash_table):
        # Perform linear interpolation along the x-axis
        x_frac = (x_val - x_idx_min) / (x_idx_max - x_idx_min)
        y_frac = (y_val - y_idx_min) / (y_idx_max - y_idx_min)
        z_frac = (z_val - z_idx_min) / (z_idx_max - z_idx_min)

        c000 = (1 - x_frac) * (1 - y_frac) * (1 - z_frac)
        c001 = (1 - x_frac) * (1 - y_frac) * z_frac
        c010 = (1 - x_frac) * y_frac * (1 - z_frac)
        c011 = (1 - x_frac) * y_frac * z_frac
        c100 = x_frac * (1 - y_frac) * (1 - z_frac)
        c101 = x_frac * (1 - y_frac) * z_frac
        c110 = x_frac * y_frac * (1 - z_frac)
        c111 = x_frac * y_frac * z_frac

        f = self.get_features_from_hash_table  # Renaming, for code simplicity

        interpolated_value = (
                c000 * f(x_idx_min, y_idx_min, z_idx_min, hash_table) +
                c001 * f(x_idx_min, y_idx_min, z_idx_max, hash_table) +
                c010 * f(x_idx_min, y_idx_max, z_idx_min, hash_table) +
                c011 * f(x_idx_min, y_idx_max, z_idx_max, hash_table) +
                c100 * f(x_idx_max, y_idx_min, z_idx_min, hash_table) +
                c101 * f(x_idx_max, y_idx_min, z_idx_max, hash_table) +
                c110 * f(x_idx_max, y_idx_max, z_idx_min, hash_table) +
                c111 * f(x_idx_max, y_idx_max, z_idx_max, hash_table)
        )

        return interpolated_value

    def get_features_from_hash_table(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, hash_table):
        # 有些人喜欢‘强复用’，我不喜欢，代码才几个KB，牺牲可读性来换取存储不值得。对于小代码量的东西，也不需要太多力气维护。
        this_string = str(x.item()) + str(y.item()) + str(z.item())
        try:
            features = hash_table[this_string]
        except:
            hash_table[this_string] = 0
            features = self.features_sample()
        return features

    def get_ray_coordinate(self, c2w, ray_direction, ray_depth):
        '''
        First calculate the relative position the point
        and then convert it to homography representation
        and then appy c2w to it
        :param c2w:
        :param ray_direction:
        :param ray_depth:
        :return:
        '''
        point2world = (c2w @ torch.cat((ray_direction * ray_depth, torch.tensor([1]))))[0:3]
        return point2world

    def features_sample(self):
        sample = torch.zeros((2))
        return sample
