import torch

# Example tensors
R_f = torch.rand(816000, 16, 9)  # Shape: [batch_size, num_matrices, matrix_size]
SH = torch.rand(816000, 9)       # Shape: [batch_size, matrix_size]

# Expand SH to have the same size as R_f along dimension 1
SH_expanded = SH.unsqueeze(1)  # Shape: [batch_size, 1, matrix_size]

# Perform batch matrix multiplication along the third dimension
test=SH_expanded.transpose(1,2)
result = torch.bmm(R_f, SH_expanded.transpose(1, 2))  # Shape: [batch_size, num_matrices, 1]

# Squeeze the result to remove the singleton dimension
result_squeezed = result.squeeze(2)  # Shape: [batch_size, num_matrices]

# Alternatively, you can achieve this directly using broadcasting
result_broadcast = torch.sum(R_f * SH_expanded, dim=2)  # Shape: [batch_size, num_matrices]

# Check if the results are the same
print(torch.allclose(result_squeezed, result_broadcast))  # Should print True