from config import sim_params
from vector_fitter_class import VectorFitter
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def view_3d_array(array):
    current_slice = 0  # Initial slice index

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Display the initial slice
    img = ax.imshow(array[current_slice, :, :], cmap='gray')
    ax.axis('off')

    # Create a slider widget for slice selection
    slider_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(slider_ax, 'Slice', 0, array.shape[0] - 1, valinit=current_slice, valstep=1)

    def update(val):
        nonlocal current_slice
        current_slice = int(slider.val)
        img.set_data(array[current_slice, :, :])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def partition_tensor(tensor, max_size):
    tensor_size = tensor.size(0)
    num_chunks = (tensor_size + max_size - 1) // max_size

    chunks = []
    for i in range(num_chunks):
        start_idx = i * max_size
        end_idx = min((i + 1) * max_size, tensor_size)
        chunk = tensor[start_idx:end_idx]
        chunks.append(chunk)

    return chunks


def create_adjusted_repeated_tensor(final_length, repeat_factor, max_frames, value_range=(0, 256)):
    random_first_column = torch.randint(0, max_frames, size=(final_length, 1))
    random_last_columns = torch.randint(value_range[0], value_range[1], size=(final_length // repeat_factor, 2))
    random_last_columns = random_last_columns.repeat((repeat_factor,1))
    repeated_tensor = torch.cat((random_first_column, random_last_columns), dim=1)


    return repeated_tensor

# Example usage
original_tensor = torch.arange(1, 101)  # Create a tensor from 1 to 100
max_chunk_size = 1000

params = sim_params(zstack=False)
params.batch_size = 20000
params.lambda_damping = 0.01

#VPSF = ##torch.compile(VectorFitter(params))
VPSF = VectorFitter(params)

numbeads = 2000
dx = (2* torch.rand((numbeads, 1))-1) * params.pixelsize
dy = (2 * torch.rand((numbeads, 1))-1) * params.pixelsize
dz = (2*  torch.rand((numbeads, 1))-1) * 300
Nphotons = torch.ones((numbeads, 1)) * 2000
Nbackground = torch.ones((numbeads, 1)) * 10
ground_truth = torch.concat((dx, dy, dz, Nphotons, Nbackground), axis=1).to(params.dev)



roipos= create_adjusted_repeated_tensor(numbeads, 100, 100)

roi_pos_nump = roipos.detach().cpu().numpy()
import numpy as np
import tifffile
np.savetxt('example_roipos.txt', roi_pos_nump)

mu, dmu = VPSF.poissonrate(ground_truth)

spots = torch.poisson(mu)
tifffile.imwrite('example_spots.tiff', spots.detach().cpu().numpy())

VPSF.fit_emitters_batched(spots, roipos)
#
# param_range = torch.tensor([
#     [-(VPSF.Mx / 2 - 2) * VPSF.pixelsize, (VPSF.Mx / 2 - 2) * VPSF.pixelsize],
#     [-(VPSF.Mx / 2 - 2) * VPSF.pixelsize, (VPSF.Mx / 2 - 2) * VPSF.pixelsize],
#     [VPSF.zspread[0], VPSF.zspread[1]],
#     [1, 1e9],
#     [0.5, 1000],
# ])
#
# def find_first_zero_indices(tensor):
#     zero_indices = (tensor == 0).nonzero()
#     result = torch.zeros(tensor.size(1), dtype=torch.int64)
#
#     for col in range(tensor.size(1)):
#         col_indices = zero_indices[zero_indices[:, 1] == col]
#         if col_indices.numel() > 0:
#             result[col] = col_indices[0, 0]
#         else:
#             result[col] = tensor.size(0) - 1
#
#     return result
# estim_list = []
# iterations_vector_tot = []
# crlb_tot = []
# batch_size = 10
# initial_guess = ground_truth * 1.2
# # Print the chunked tensors
# chunked_spots = partition_tensor(spots, batch_size)
# chunked_init_guess = partition_tensor(initial_guess,batch_size)
# for idx, smp in enumerate(chunked_spots):
#     estim, traces = VPSF.LM_MLE(smp, param_range,chunked_init_guess[idx])
#     mu, jac = VPSF.poissonrate(estim)
#     crlb = VPSF.compute_crlb(mu,jac)
#     estim_list.append(estim)
#     iterations_vector = find_first_zero_indices(traces[:,:,-1])
#     iterations_vector_tot.append(iterations_vector)
#     crlb_tot.append(crlb)
# estim_final = torch.cat(estim_list)
# estim_final[:,0:2] = estim_final[:,0:2] /params.pixelsize + params.Mx/2
# iterations_final = torch.cat(iterations_vector_tot).to(params.dev)
# crlb_final = torch.cat(crlb_tot)
# border = 2.1
# sel_pos = ((estim_final[:, 0] > border) & (estim_final[:, 0] < params.Mx - border - 1) &
#        (estim_final[:, 1] > border) & (estim_final[:, 1] < params.Mx - border - 1) &
#        (estim_final[:, 2] > params.zspread[0]) & (estim_final[:, 2] < params.zspread[1]))
#
# sel = torch.logical_and(sel_pos, (iterations_final < params.Nitermax))
#
# print(
#     f'Filtering on position in ROI: {estim_final.size(0) - sel_pos.sum()}/{estim_final.size(0)} spots removed.\n'
#     f'Filtering on iterations : {estim_final.size(0) - (iterations_final < params.Nitermax).sum()}/{estim_final.size(0)} spots removed.')




#view_3d_array(mu.detach().cpu().numpy())
