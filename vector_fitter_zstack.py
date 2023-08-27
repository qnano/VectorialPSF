from config import z_stack_params
import tifffile
import numpy as np
import napari
import sys

def show_napari(img):
    import napari
    with napari.gui_qt():
        viewer = napari.view_image(img)

from glrt import torch_stuff

import torch

params = z_stack_params()

data = tifffile.imread('bead.tif')
data = data[None,...]
roi_pos = np.zeros(4)


wavevector,wavevectorzimm,Waberration,all_zernikes,PupilMatrix = get_pupil_matrix(params)

theta_test,theta_zstack = initial_guess(data)
Matlab = False
# matlab comparison
if Matlab:
    theta_zstack = np.array((26.6775710391987, 39.7029110767899, 0, 7535.01250000000, 5))[None, ...]
else:
    # good  initial values
    zmin = params.zrange[0]
    zmax = params.zrange[1]
    Kz = params.K
    K = Kz * 1
    dz = (zmax - zmin) / Kz
    Zimage = np.linspace(zmin, zmax + dz, Kz)
    params.xemit = (theta_zstack[0, 0] - params.Mx / 2) * params.pixelsize  # distance from center
    params.yemit = (theta_zstack[0, 1] - params.My / 2) * params.pixelsize
    params.zemit = Zimage[int(theta_zstack[0, 2])]
    theta_zstack[0,2] = Zimage[int(theta_zstack[0, 2])]



theta = np.zeros(params.numparams)
theta[0:5] = theta_zstack
theta[6] = 100
theta[3] = 4000
theta[4] = 10
thetamin, thetamax = thetalimits(params, theta)


spots = data[0]
spots = np.transpose(spots, [1, 2, 0]) * 1
thetaretry = (thetamax + thetamin) / 2

mu, dmudtheta = poissonrate(params, theta, PupilMatrix, all_zernikes, wavevector, wavevectorzimm)
merit,grad,Hessian = likelihood(params,spots,mu,dmudtheta)

# pre - allocate
thetatemp = np.zeros((params.numparams, params.Nitermax + 1))
merittemp = np.zeros((1, params.Nitermax + 1))
thetatemp[:, 0] = theta
merittemp[0] = merit
meritprev = merit*1

tollim = params.tollim
itter = 1
monitor = 2 * tollim
alamda = 1e-2
alamdafac = 10

while ((itter < params.Nitermax) and (monitor > tollim)):
    print(itter)
    print(monitor)
    matty = Hessian + alamda * np.diag(np.diag(Hessian))

    # thetaupdate
    # update of fit parameters via Levenberg-Marquardt
    Bmat = Hessian + alamda * np.diag(np.diag(Hessian))
    dtheta = np.linalg.solve(-Bmat,grad.T)
    thetatry= theta+dtheta.T
    # enforce physical boundaries in parameter space.
    for jj in range(len(theta)):
        if (thetatry[jj] > thetamax[jj]) or (thetatry[jj] < thetamin[jj]):
            thetatry[jj] = thetaretry[jj]*1

    mu, dmudtheta = poissonrate(params,thetatry,PupilMatrix,all_zernikes,wavevector,wavevectorzimm)
    [merittry, gradtry, Hessiantry] = likelihood(params, spots, mu, dmudtheta)
    dmerit = merittry - merit

    # modify Levenberg-Marquardt parameter
    if (dmerit < 0):
        alamda = alamdafac * alamda
    else:
        alamda = alamda / alamdafac
        theta = thetatry*1
        merit = merittry*1
        grad = gradtry*1
        Hessian = Hessiantry*1
        dmerit = merit - meritprev
        monitor = abs(dmerit / merit)
        meritprev = merit
        thetaretry = theta

    
    itter += 1






#
# param_range = np.concatenate((thetamin[..., None], thetamax[..., None]), axis=-1)
#
# param_range = torch.tensor(param_range)
# model = Vector_psf(params, PupilMatrix, all_zernikes, wavevector, wavevectorzimm)
# mle = LM_MLE(model, lambda_=1e-6, iterations=200,
#               param_range_min_max=param_range)
#
# initial = np.zeros(params.numparams)
# initial[0:5] = theta_zstack
# dev = 'cuda'
# smp_ = torch.tensor(data)
#
# #
# # if multiple_gpu:
# #     mle = torch.nn.DataParallel(mle)  # select if multiple gpus
# # else:
# #     mle = torch.jit.script(mle)  # select if single gpus
# params_, loglik_, _ = mle.forward(smp_, torch.Tensor(initial).to(dev))