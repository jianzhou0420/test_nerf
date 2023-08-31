import torch
import torch.nn as nn

# 好，到现在为止，数据已经可以读取了，tracking不做的情况下，下面需要的就是最重要的mapping的部分，用explicit的方法来mapping
# batch={dict:5}
#   'frame_id'=Tensor(1,)
#   'c2w'=Tensor(1,4,4)
#   'rgb'=Tensor(1,680,1200,3)
#   'depth'=Tensor(1,680,1200)
#   'direction'=Tensor(1,680,1200,3)
# test
# 这个batch用来让IDE识别batch里面到底是什么，有助于我写代码
batch = {'frame_id': torch.zeros(1),
         'c2w': torch.zeros((1, 4, 4)),
         'rgb': torch.zeros((1, 680, 1200, 3)),
         'depth': torch.zeros((1, 680, 1200)),
         'direction': torch.zeros((1, 680, 1200, 3))}
# /test
# 先算
print(batch['depth'].numel())
test = batch['depth'][0]

batch['frame_id'] = batch['frame_id'].squeeze()
batch['c2w'] = batch['c2w'].squeeze()
batch['rgb'] = torch.flatten(batch['rgb'], 0, 2)
batch['depth'] = torch.flatten(batch['depth'])
batch['direction'] = torch.flatten(batch['direction'], 0, 2)

# Now batch={dict:5}
#   'frame_id'=Tensor(1)
#   'c2w'=Tensor(4,4)
#   'rgb'=Tensor(680*1200,3)
#   'depth'=Tensor(680*1200)
#   'direction'=Tensor(680*1200,3)


batch['c2w'] = torch.tensor((
    [1, 0, 0, 1],
    [0, 0, -1, 1],
    [0, 1, 0, 1],
    [0, 0, 0, 1]), dtype=torch.float32)
batch['depth'][0] = 10
batch['direction'][0] = torch.tensor((0, 0, 1), dtype=torch.float32)

grid_steps_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
bound = [[-3, 3, -1, 1, -1, 1]]
grid_list = []
for i in range(len(grid_steps_list)):
    grid_list.append(dict())


def trilinear_interpolation(x_idx_min, y_idx_min, z_idx_min,
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

    f = get_features_from_hash_table  # Renaming, for code simplicity

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


def get_features_from_hash_table(x: torch.Tensor, y, z, hash_table):
    # 有些人喜欢‘强复用’，我不喜欢，代码才几个KB，牺牲可读性来换取存储不值得。对于小代码量的东西，也不需要太多力气维护。
    this_string = str(x.item()) + str(y.item()) + str(z.item())
    try:
        features = hash_table[this_string]
    except:
        hash_table[this_string] = 0
        features = 0  # T
    return features


def get_ray_coordinate(c2w, ray_direction, ray_depth):
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


# 先算
for i in range(batch['depth'].numel()):  # depth can get the number of items
    c2w = batch['c2w']
    ray_direction = batch['direction'][i]
    ray_depth = batch['depth'][i]
    point_location = get_ray_coordinate(c2w, ray_direction, ray_depth)

    features = []

    for j in range(len(grid_steps_list)):
        x_idx_min = point_location[0] // grid_steps_list[j]
        y_idx_min = point_location[1] // grid_steps_list[j]
        z_idx_min = point_location[2] // grid_steps_list[j]
        x_idx_max = x_idx_min + 1
        y_idx_max = y_idx_min + 1
        z_idx_max = z_idx_min + 1

        this_features = trilinear_interpolation(x_idx_min, y_idx_min, z_idx_min,
                                                x_idx_max, y_idx_max, z_idx_max,
                                                point_location[0], point_location[1], point_location[2],
                                                grid_list[j])
        features.append(this_features)
        print(1)
    # now we have the location of the point, let's find their neighbour vertices

    break


########################################################################################################################
def features_sample(self):
    sample = torch.zeros((4))  # TODO: replace it to real sample
    return sample


feature_size = features_sample().size()
grid_size = len(grid_steps_list)
all_batch_features = torch.zeros((680 * 1200, grid_size, feature_size))


##好，我以及query了hash grid了，如何？还有network的forward。跟随在

class NeRFmap(torch.nn.Module):
    def __init__(self):
        super(NeRFmap,self).__init__()
        model=nn.Sequential(
            nn.Linear(grid_size,256),
            nn.Linear(256,256),
            nn.Linear(256,256),
        )
        pass
    def forward(self,all_batch_feature):



class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(16*2, hidden_size)   # 16 layers,
        self.relu = nn.ReLU()                          # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def grid_hash(pos_grid,HASH_TYPE):
    if HASH_TYPE=='Prime':
        return prime_hash(pos_grid)
    elif HASH_TYPE=='CoherentPrime':
        return  coherent_prime_hash(pos_grid)
    elif HASH_TYPE=='ReversePrime':
        return reverse_prime_hash(pos_grid)
    else: TypeError()

def apply_hash(pos_grid,factors):
    assert len(pos_grid)<=len(factors), 'dimensions of pos_grid exceeding to the size of factors'
    result=0
    for i in range(len(pos_grid)):
        result ^= pos_grid[i] *factors[i]


def prime_hash(pos_grid):
    factors=list([1958374283, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737])
    result=apply_hash(pos_grid,factors)
    return result
def coherent_prime_hash(pos_grid):
    factors=list([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737])
    result=apply_hash(pos_grid,factors)
    return result
def reverse_prime_hash(pos_grid):
    factors=list([2165219737, 1434869437, 2097192037, 3674653429, 805459861, 2654435761, 1958374283])
    result=apply_hash(pos_grid,factors)
    return result

# template <uint32_t N_DIMS>
# __device__ uint32_t prime_hash(const uvec<N_DIMS>& pos_grid) {
#     constexpr uint32_t factors[7] = { 1958374283u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };
# return lcg_hash<N_DIMS, 7>(pos_grid, factors);
# }
#
# template <uint32_t N_DIMS>
#                    __device__ uint32_t coherent_prime_hash(const uvec<N_DIMS>& pos_grid) {
# constexpr uint32_t factors[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };
# return lcg_hash<N_DIMS, 7>(pos_grid, factors);
# }
#
# template <uint32_t N_DIMS>
#                    __device__ uint32_t reversed_prime_hash(const uvec<N_DIMS>& pos_grid) {
# constexpr uint32_t factors[7] = { 2165219737u, 1434869437u, 2097192037u, 3674653429u, 805459861u, 2654435761u, 1958374283u };
# return lcg_hash<N_DIMS, 7>(pos_grid, factors);
# }
