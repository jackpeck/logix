import torch
import einops

# (Pdb) preconditioned_grads['model.layers.19.mlp.up_proj']['grad'].shape
# torch.Size([1, 6144, 2048])

torch.manual_seed(0)
d_model = 2048
d_mlp = 6144
gradient_matrix = torch.randn(d_mlp, d_model)
# print(gradient_matrix)


dict_size = 4
encoder_row_rank = 32
project_sketch_size_in = 100
project_sketch_size_out = 100
encoder_u = torch.randn(4, d_mlp, encoder_row_rank)
encoder_v = torch.randn(4, d_model, encoder_row_rank)

project_in = torch.randn(project_sketch_size_in, d_model)
project_out = torch.randn(project_sketch_size_in, d_mlp)

project_temp_left = einops.einsum(project_in, encoder_v, 'si d_model, q d_model r -> q r si')
project_temp_right = einops.einsum(project_out, encoder_u, 'so d_mlp, q d_mlp r -> q r so')
# breakpoint()

projected_encoder_rows_non_flattened = einops.einsum(project_temp_left, project_temp_right, 'q r si, q r so -> q si so')
projected_encoder_rows = einops.rearrange(projected_encoder_rows_non_flattened, 'q si so -> q (si so)')

projected_gradient_non_flattened = project_out @ gradient_matrix @ project_in.T
projected_gradient = einops.rearrange(projected_gradient_non_flattened, 'si so -> (si so)')

breakpoint()
