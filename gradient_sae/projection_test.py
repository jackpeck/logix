import torch
import einops

# (Pdb) preconditioned_grads['model.layers.19.mlp.up_proj']['grad'].shape
# torch.Size([1, 6144, 2048])

torch.manual_seed(0)
d_model = 2048
d_mlp = 6144
batch_size = 8
gradient_matrix = torch.randn(batch_size, d_mlp, d_model)
# print(gradient_matrix)


dict_size = 4
encoder_row_rank = 32
project_sketch_size_in = 100
project_sketch_size_out = 50
encoder_u = torch.randn(dict_size, d_mlp, encoder_row_rank)
encoder_v = torch.randn(dict_size, d_model, encoder_row_rank)

project_in = torch.randn(project_sketch_size_in, d_model) / (project_sketch_size_in ** 0.5)
project_out = torch.randn(project_sketch_size_out, d_mlp) / (project_sketch_size_out ** 0.5)

project_temp_left = einops.einsum(project_in, encoder_v, 'si d_model, q d_model r -> q r si')
project_temp_right = einops.einsum(project_out, encoder_u, 'so d_mlp, q d_mlp r -> q r so')
# breakpoint()


projected_encoder_rows_non_flattened = einops.einsum(project_temp_left, project_temp_right, 'q r si, q r so -> q si so')
projected_encoder_rows = einops.rearrange(projected_encoder_rows_non_flattened, 'q si so -> q (si so)')

projected_gradient_non_flattened = project_out @ gradient_matrix @ project_in.T
projected_gradient = einops.rearrange(projected_gradient_non_flattened, 'b si so -> b (si so)')



approx_sat_latent_presparsification = einops.einsum(projected_gradient, projected_encoder_rows, 'b siso, q siso -> b q')


approx_search_threshold_proportion = 0.2
approx_search_threshold = torch.quantile(approx_sat_latent_presparsification, 1 - approx_search_threshold_proportion)


approx_sat_latent_presparsification >= approx_search_threshold

breakpoint()


# should be equiv to this dot product ((encoder_u[2] @ encoder_v[2].T) * gradient_matrix[1]).sum()
trUTGV = encoder_u[2].T @ gradient_matrix[1] @ encoder_v[2] # <UV^T, G>_F = Tr(U^T G V)
einops.einsum(trUTGV, 'r r -> ') # note: folding trace into above may be faster


x = einops.einsum(encoder_u, gradient_matrix, encoder_v, 'q d_mlp ru, b d_mlp d_model, q d_model rv -> b q ru rv')
einops.einsum(x, 'b q r r -> b q')

unfinished, possibly a bug in siso ordering