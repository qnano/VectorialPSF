import matplotlib
import ypstruct as yp
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from vector_fitter_class import VectorFitter
import torch
#matplotlib.use('Qt5Agg')
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


# Start off with the Zernikes up to j=15
_noll_n = [0,0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]
_noll_m = [0,0,1,-1,0,-2,2,-1,1,-3,3,0,2,-2,4,-4]
def noll_to_zern(j):
    """Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike
    index.

    c.f. https://oeis.org/A176988

    Parameters:
        j:      Zernike mode Noll index

    Returns:
        (n, m) tuple of Zernike indices
    """
    while len(_noll_n) <= j:
        n = _noll_n[-1] + 1
        _noll_n.extend( [n] * (n+1) )
        if n % 2 == 0:
            _noll_m.append(0)
            m = 2
        else:
            m = 1
        # pm = +1 if m values go + then - in pairs.
        # pm = -1 if m values go - then + in pairs.
        pm = +1 if (n//2) % 2 == 0 else -1
        while m <= n:
            _noll_m.extend([ pm * m , -pm * m ])
            m += 2

    return _noll_n[j], _noll_m[j]


def api_gui(gui_inputs):

    params = yp.struct()
    params.debugmode = False
    # fitting parameters
    params.Nitermax = int(gui_inputs['Nitermax'])
    params.tollim = float(gui_inputs['tolerance'])
    params.zstack = int(gui_inputs['z_stack'])
    params.zstack = bool(gui_inputs['z_stack'])

    # camera offset and gain assume is calibrated
    params.offset = 0
    params.gain = 1


    params.dev = gui_inputs['dev']

    # PSF/optical parameters
    params.NA = float(gui_inputs['NA'])
    params.refmed = float(gui_inputs['refmed'])
    params.refcov = float(gui_inputs['refcov'])
    params.refimm = float(gui_inputs['refimm'])
    params.refimmnom =params.refimm

    params.fwd = float(gui_inputs['fwd'])##

    params.depth = -float(gui_inputs['depth']) # this minus is to switch coordinate system of z


    params.zspread = [float(gui_inputs['zspread[0]']),float(gui_inputs['zspread[1]'])] # spread - limits of estimator
    if not params.zstack:
        params.zrange = [-float(gui_inputs['zspread[0]']),
                         -float(gui_inputs['zspread[1]'])]# actual range - only relevant for zstack
    else:
        params.zrange = [-float(gui_inputs['zrange[0]']),
                        -float(gui_inputs['zrange[1]'])]  #
    #params.ztype = 'stage' # 'medium'
    params.Lambda = float(gui_inputs['Lambda'])
    params.imgwidth = int(gui_inputs['width'])
    params.imgheight = int(gui_inputs['height'])
    params.lambdacentral = params.Lambda
    params.lambdaspread = [params.Lambda,params.Lambda]
    params.xemit = 0.0
    params.yemit = 0.0
    params.zemit = 0.0
    params.Npupil = int(gui_inputs['Npupil'])
    params.lambda_damping = float(gui_inputs['lambda_damping'])
    params.batch_size = int(gui_inputs['batch_size'])
    params.pixelsize = float(gui_inputs['pixelsize'])
    params.samplingdistance = params.pixelsize
    params.Mx = int(gui_inputs['K']) # roisize
    params.My = params.Mx
    params.Mz = 1
    params.dev = gui_inputs['dev']
    params.xrange = params.pixelsize*params.Mx/2
    params.yrange = params.pixelsize*params.My/2
    # model parameters
    params.alpha = 0
    params.beta = 0
    params.K = 1  # zstack number
    params.m = 1
    if gui_inputs.get('aberrations') is not None:
        num_aberrations = np.size(gui_inputs['aberrations'])
        aberrations = gui_inputs['aberrations']

    else:
        num_aberrations = int(gui_inputs['numaber'])
        aberrations = np.zeros((num_aberrations, 3))
        for index in range(num_aberrations):
            nol_index = index + 5  # we start at 5
            m, n = noll_to_zern(nol_index)
            aberrations[index, :] = np.array([m, n, 0])
    print(aberrations)
    params.aberrations = aberrations
    params.aberrations[:, 2] = params.aberrations[:, 2]

    # SAF check
    if params.NA>params.refmed and abs(params.depth)<2*params.Lambda:
        zvals, _ = set_saffocus(params)
    else:
        zvals, _ = get_rimismatch(params)

    params.zvals = zvals


    PupilSize = params.NA / params.Lambda
    params.numparams = 5 + np.size(params.aberrations[:, 2]) # only x,y,z,N,Nbg
    # calculate auxiliary vectors for chirpz

    Ax,Bx,Dx = prechirpz(PupilSize, params.xrange, params.Npupil, params.Mx)
    Ay, By, Dy = prechirpz(PupilSize, params.yrange, params.Npupil, params.My)

    params.cztN = params.Npupil
    params.cztM = params.Mx
    params.cztL = params.Npupil + params.Mx - 1

    params.Axmt = np.matlib.repmat(Ax, params.Mx, 1)
    params.Bxmt = np.matlib.repmat(Bx, params.Mx, 1)
    params.Dxmt = np.matlib.repmat(Dx, params.Mx, 1)
    params.Aymt = np.matlib.repmat(Ay, params.Npupil, 1)
    params.Bymt = np.matlib.repmat(By, params.Npupil, 1)
    params.Dymt = np.matlib.repmat(Dy, params.Npupil, 1)


    return params, zvals

def get_rimismatch(params):

    refins = np.array([params.refimm, params.refimmnom, params.refmed])
    zvals = np.array([0, params.fwd ,-params.depth])
    NA = params.NA
    K = len(refins)
    if (NA>params.refmed):
        NA = params.refmed

    fsqav = np.zeros((K,1))
    fav =  np.zeros((K,1))
    Amat =  np.zeros((K,K))


    for jj in range(K):
        fsqav[jj] = refins[jj]**2 - 0.5 * NA**2
        fav[jj] = ((2/3)/NA**2) * (refins[jj]**3 - (refins[jj]**2-NA**2)**(3/2))
        Amat[jj, jj] = fsqav[jj] - fav[jj]**2

        for kk in range(jj):
            Amat[jj,kk] = (1/4/NA**2) * (refins[jj] * refins[kk] * (refins[jj]**2 + refins[kk]**2) -
            (refins[jj]**2 + refins[kk]**2 - 2*NA**2) * np.sqrt(refins[jj]**2-NA**2) * np.sqrt(refins[kk]**2-NA**2) +
            (refins[jj]**2 - refins[kk]**2)**2*np.log((np.sqrt(refins[jj]**2 - NA**2)+np.sqrt(refins[kk]**2-NA**2))/(refins[jj]+refins[kk])))
            Amat[jj, kk] = Amat[jj,kk] - fav[jj] * fav[kk]
            Amat[kk,jj] = Amat[jj,kk]*1
    zvalsratio = np.zeros((K, 1))
    Wrmsratio = np.zeros((K, K))
    for jvpr in range(K-1):
        jv = jvpr + 1
        zvalsratio[jv] = Amat[0,jv]/Amat[0,0]
        for kvpr in range(K-1):
            kv = kvpr + 1

            Wrmsratio[jv,kv] = Amat[jv,kv] - Amat[0, jv]*Amat[0,kv]/Amat[0,0]

    zvals[0] = zvalsratio[1] * zvals[1] + zvalsratio[2] * zvals[2]
    Wrms = Wrmsratio[1,1] * zvals[1] ** 2 + Wrmsratio[2,2] * zvals[2] ** 2 + 2 * Wrmsratio[1,2]* zvals[1] * zvals[2]
    Wrms = np.sqrt(Wrms)

    return zvals, Wrms

def set_saffocus(params):
    NA = params.NA
    refmed = params.refmed
    refcov = params.refcov
    refimm = params.refimm
    refimmnom = params.refimmnom
    Lambda = params.Lambda
    Npupil = params.Npupil

    zvals = np.array([0, params.fwd, -params.depth])

    # pupil radius (in diffraction units) and pupil coordinate sampling
    PupilSize = 1.0
    DxyPupil = 2*PupilSize/Npupil
    XYPupil = np.arange(-PupilSize+DxyPupil/2,PupilSize,DxyPupil)
    YPupil,XPupil = np.meshgrid(XYPupil,XYPupil)

    # % calculation of relevant Fresnel-coefficients for the interfaces
    # % between the medium and the cover slip and between the cover slip
    # % and the immersion fluid
    # % The Fresnel-coefficients should be divided by the wavevector z-component
    # % of the incident medium, this factor originates from the
    # % Weyl-representation of the emitted vector spherical wave of the dipole.

    argMed = 1-(XPupil**2+YPupil**2)*NA**2/refmed**2
    phiMed = np.arctan2(0,argMed)
    CosThetaMed = np.sqrt(abs(argMed))*(np.cos(phiMed/2)-1j*np.sin(phiMed/2) - 0j)
    CosThetaCov = np.sqrt(1-(XPupil**2+YPupil**2)*NA**2/refcov**2 - 0j)
    CosThetaImm = np.sqrt(1-(XPupil**2+YPupil**2)*NA**2/refimm**2- 0j)
    CosThetaImmnom = np.sqrt(1-(XPupil**2+YPupil**2)*NA**2/refimmnom**2- 0j)

    FresnelPmedcov = 2*refmed*CosThetaMed/(refmed*CosThetaCov+refcov*CosThetaMed)
    FresnelSmedcov = 2*refmed*CosThetaMed/(refmed*CosThetaMed+refcov*CosThetaCov)
    FresnelPcovimm = 2*refcov*CosThetaCov/(refcov*CosThetaImm+refimm*CosThetaCov)
    FresnelScovimm = 2*refcov*CosThetaCov/(refcov*CosThetaCov+refimm*CosThetaImm)
    FresnelP = FresnelPmedcov*FresnelPcovimm
    FresnelS = FresnelSmedcov*FresnelScovimm

    # setting of vectorial functions
    Phi = np.arctan2(YPupil,XPupil)
    CosPhi = np.cos(Phi)
    SinPhi = np.sin(Phi)
    CosTheta = CosThetaMed #sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refmed^2);
    SinTheta = np.sqrt(1-CosTheta**2)

    pvec = np.zeros((Npupil, Npupil,3), dtype=complex)
    svec = np.zeros((Npupil, Npupil,3), dtype=complex)

    pvec[:,:,0] = (FresnelP+0j)*(CosTheta+0j)*(CosPhi+0j)
    pvec[:,:,1] = FresnelP*CosTheta*(SinPhi+0j)
    pvec[:,:,2] = -FresnelP*(SinTheta+0j)
    svec[:,:,0]  = -FresnelS*(SinPhi+0j)
    svec[:,:,1]  = FresnelS*(CosPhi +0j)
    svec[:,:,2]  = 0

    PolarizationVector = np.zeros((Npupil,Npupil,2,3), dtype=complex)
    PolarizationVector[:,:,0,:] = CosPhi[:,:,None]*pvec-SinPhi[:,:,None]*svec
    PolarizationVector[:,:,1,:] = SinPhi[:,:,None]*pvec+CosPhi[:,:,None]*svec

    # definition aperture
    ApertureMask = (XPupil**2+YPupil**2)<1.0

    ##
    PupilSize = 1.0
    DxyPupil = 2 * PupilSize / params.Npupil
    XYPupil = np.arange(-PupilSize + DxyPupil / 2, PupilSize, DxyPupil)
    YPupil, XPupil = np.meshgrid(XYPupil, XYPupil)
    size_pup = np.shape(XPupil)
    Waberration = np.zeros(size_pup)
    orders = params.aberrations[:, 0:2]
    zernikecoefs = params.aberrations[:, 2]
    normfac = np.sqrt(2 * (orders[:, 0] + 1) / (1 + (orders[:, 1] == 0)))
    zernikecoefs = normfac * zernikecoefs
    all_zernikes = get_zernikefunctions(orders, XPupil, YPupil)
    for j in range(len(zernikecoefs)):
        Waberration = Waberration + zernikecoefs[j] * all_zernikes[:, :, j]
    ##

    # aplanatic amplitude factor
    Amplitude = ApertureMask*np.sqrt(CosThetaImm)/(refmed*CosThetaMed)
    Strehlnorm = 0
    for itel in range(2):
        for jtel in range(3):
            Strehlnorm = Strehlnorm + abs(np.sum(np.sum(Amplitude*np.exp(2*np.pi*1j*np.real(Waberration))*(PolarizationVector[:,:,itel,jtel]))))**2

    Nz = 1001
    zpos = np.linspace(3/2*-300,3/2*300,Nz)
    Strehl = np.zeros((Nz))
    for jz in range(Nz):
        zvals[0] = params.fwd - 1.25*refimm/refmed*params.depth + zpos[jz]
        Wzpos = zvals[0]*refimm*CosThetaImm-zvals[1]*refimmnom*CosThetaImmnom-zvals[2]*refmed*CosThetaMed
        Wzpos = Wzpos*ApertureMask
        for itel in range(2):
            for jtel in range(3):
                Strehl[jz] = Strehl[jz] + abs(np.sum(np.sum(Amplitude*np.exp(2*np.pi*1j*np.real(Wzpos+Waberration)/Lambda)*PolarizationVector[:,:,itel,jtel])))**2/Strehlnorm

    indz = np.argmax(Strehl)

    if indz <= 2:
        indz = 2
    elif indz > (Nz-3):
        indz = Nz-3

    zfit = np.polyfit(zpos[indz - 2:indz + 2], Strehl[indz - 2:indz + 2], 2)
    zvals[0] = params.fwd - 1.25 * refimm / refmed * params.depth - zfit[1] / (2 * zfit[0])
    MaxStrehl = np.polyval(zfit,- zfit[1]/(2*zfit[0]))
    Wrms = Lambda/(2*np.pi)*np.log(1/MaxStrehl)

    if params.debugmode:
        print('image plane depth from cover slip = %f nm\n' % -zvals[2])
        print('free working distance = %f mu\n' % (1e-3*zvals[1]))
        print('nominal z-stage position = %f mu\n' % (1e-3 * zvals[0]))
        print('rms aberration due to RI mismatch = %f mlambda\n' % (1e3 * Wrms / params.Lambda))

        plt.figure()
        plt.plot(zpos - 1.25 * refimm / refmed * params.depth, Strehl)
        plt.xlabel('Stage position')
        plt.ylabel('Strehl')
        plt.show()

    return zvals, Wrms

def prechirpz(xsize,qsize,N,M):


    L = N+M-1
    sigma = 2*np.pi*xsize*qsize/N/M
    Afac = np.exp(2*1j*sigma*(1-M))
    Bfac = np.exp(2*1j*sigma*(1-N))
    sqW = np.exp(2*1j*sigma)
    W = sqW**2
    Gfac = (2*xsize/N)*np.exp(1j*sigma*(1-N)*(1-M))

    Utmp = np.zeros((N),dtype=complex)
    A = np.zeros((N),dtype=complex)
    Utmp[0] = sqW*Afac
    A[0]= 1.0

    START = 1
    for index, item in enumerate(Utmp[START:], START):
        A[index] = Utmp[index-1]*A[index-1]
        Utmp[index] = Utmp[index - 1] * W


    Utmp = np.zeros(M,dtype=complex)
    B = np.ones(M,dtype=complex)
    Utmp[0] = sqW*Bfac
    B[0] = Gfac
    for index, item in enumerate(Utmp[START:], START):
        B[index] = Utmp[index-1]*B[index-1]
        Utmp[index] = Utmp[index - 1] * W


    Utmp = np.zeros(max(N,M)+1,dtype=complex)
    Vtmp = np.zeros(max(N,M)+1,dtype=complex)
    Utmp[0] = sqW
    Vtmp[0] = 1.0

    for index, item in enumerate(Utmp[START:], START):
        Vtmp[index] = Utmp[index-1]*Vtmp[index-1]
        Utmp[index] = Utmp[index - 1] * W

    D = np.ones(L, dtype=complex)
    for i in range(M):
        D[i] = np.conj(Vtmp[i])

    for i in range(N):
        D[L-1-i] = np.conj(Vtmp[i+1])


    D = np.fft.fft(D)

    return A,B,D

def fit_aberrations(gui_inputs, beads):

    params,_ = api_gui(gui_inputs)
    beads = torch.tensor(beads).to(params.dev)
    if params.zstack == False:
        params.zstack = True
    #VPSF = torch.compile(VectorFitter(params))
    VPSF = VectorFitter(params)
    final, theta, traces = VPSF.fit_zstack(beads)
    return final.detach().cpu().numpy(), theta.detach().cpu().numpy(),traces.detach().cpu().numpy()

def fit_emitters(gui_inputs, emitters, roipos, savefn):
    params, _ = api_gui(gui_inputs)
    spots = torch.tensor(emitters).to(params.dev)
    roipos = torch.tensor(roipos).to(params.dev)
    if params.zstack == True:
        params.zstack = False
    VPSF =VectorFitter(params)

    mushow,spotsshow, s = VPSF.fit_emitters_batched(spots,roipos,savefn)

    return mushow,spotsshow,s


def make_zstack_simulation(gui_inputs, photons = 4000, bg =10, poisson_noise = False, numz=10):

    params,_ = api_gui(gui_inputs)
    flag = 0
    # if params.zstack == True:
    #     flag = 1
    #     params.zstack = False # we dont need this for the simulation
    VPSF = VectorFitter(params)

    numbeads = int(abs(params.zspread[1] - params.zspread[0])/numz)
    dx = (0 * torch.rand((numbeads, 1))) * params.pixelsize
    dy = (0* torch.rand((numbeads, 1))) * params.pixelsize
    dz = torch.linspace(params.zspread[0], params.zspread[1], numbeads)[...,None]

    Nphotons = torch.ones((numbeads, 1)) * photons
    Nbackground = torch.ones((numbeads, 1)) * bg
    ground_truth = torch.concat((dx, dy, dz, Nphotons, Nbackground), axis=1).to(params.dev)

    mu, dmu = VPSF.poissonrate(ground_truth)

    crlb = VPSF.compute_crlb(mu,dmu[...,0:5]) # only for x,y,z,N,bg
    if poisson_noise:
        mu = torch.poisson(mu)
    if flag == 1:
        params.zstack = True
    return mu, dmu, params.zspread[0], params.zspread[1], VPSF.Waberration, VPSF.PupilMatrix, crlb, dz.detach().cpu().numpy()



def get_zernikefunctions(orders, XPupil, YPupil):
    x = XPupil * 1
    y = YPupil * 1

    zersize = orders.shape
    Nzer = zersize[0]
    radormax = np.max(orders[:, 0])
    azormax = np.max(np.abs(orders[:, 1]))
    Nx, Ny = x.shape[0], x.shape[0]

    # Evaluation of the radial Zernike polynomials using the recursion relation for
    # the Jacobi polynomials.

    zerpol = np.zeros((Nx, Ny, int(radormax * azormax - 1), int(azormax) + 1))
    rhosq = x ** 2 + y ** 2
    rho = np.sqrt(rhosq)
    zerpol[:, :, 0, 0] = np.ones(x.shape)

    for jm in range(int(azormax) + 1):
        m = jm * 1
        if m > 0:
            zerpol[:, :, jm, jm] = rho * zerpol[:, :, jm - 1, jm - 1]

        zerpol[:, :, jm + 2, jm] = ((m + 2) * rhosq - m - 1) * np.squeeze(zerpol[:, :, jm, jm])
        itervalue = int(radormax) - 1 - m + 2

        for p in range(itervalue):
            piter = p + 2
            n = m + 2 * piter
            jn = n * 1

            zerpol[:, :, jn, jm] = (2 * (n - 1) * (n * (n - 2) * (2 * rhosq - 1) - m ** 2) * zerpol[:, :, jn - 2,
                                                                                             jm] -
                                    n * (n + m - 2) * (n - m - 2) * zerpol[:, :, jn - 4, jm]) / (
                                           (n - 2) * (n + m) * (n - m))

    phi = np.arctan2(y, x)

    allzernikes = np.zeros((Nx, Ny, Nzer))

    for j in range(Nzer):
        n = int(orders[j, 0])
        m = int(orders[j, 1])
        if m >= 0:
            allzernikes[:, :, j] = zerpol[:, :, n, m] * np.cos(m * phi)
        else:
            allzernikes[:, :, j] = zerpol[:, :, n, -m] * np.sin(-m * phi)

    return allzernikes