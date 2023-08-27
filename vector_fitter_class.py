import torch
import copy

class VectorFitter(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dev = params.dev
        self.zstack = params.zstack
        # region get values from params
        self.NA = params.NA
        self.refmed = params.refmed
        self.refcov = params.refcov
        self.refimm = params.refimm
        self.refimmnom = params.refimmnom
        self.damping_lm = params.lambda_damping
        self.batch_size = params.batch_size
        self.Lambda = params.Lambda
        self.Npupil = params.Npupil
        self.aberrations = torch.from_numpy(params.aberrations).to(self.dev)
        self.imgwidth = params.imgwidth
        self.imgheight = params.imgheight
        self.zvals = torch.from_numpy(params.zvals).to(self.dev)
        self.zspread = torch.tensor(params.zspread).to(self.dev)
        self.spatial_dims = 3 # 3D
        self.zrange = torch.tensor(params.zrange).to(self.dev)
        self.zmin = self.zrange[0]
        self.zmax = self.zrange[1]
        self.numparams = params.numparams
        self.numparams_fit = self.numparams * 1
        self.K = params.K
        self.Mx = params.Mx
        self.My = params.My
        self.Mz = params.Mz

        self.pixelsize = params.pixelsize
        self.Ax = torch.tensor(params.Axmt).to(self.dev)
        self.Bx = torch.tensor(params.Bxmt).to(self.dev)
        self.Dx = torch.tensor(params.Dxmt).to(self.dev)
        self.Ay = torch.tensor(params.Aymt).to(self.dev)
        self.By = torch.tensor(params.Bymt).to(self.dev)
        self.Dy = torch.tensor(params.Dymt).to(self.dev)

        self.N = params.cztN
        self.M = params.cztM
        self.L = params.cztL

        self.Nitermax = params.Nitermax
        self.tollim = params.tollim


    def get_zernikefunctions(self, orders, XPupil, YPupil):
        x = XPupil * 1
        y = YPupil * 1

        zersize = orders.size()
        Nzer = zersize[0]
        radormax = torch.max(orders[:, 0])
        azormax = torch.max(abs(orders[:, 1]))
        Nx, Ny = x.size()[0], x.size()[0]

        # Evaluation of the radial Zernike polynomials using the recursion relation for
        # the Jacobi polynomials.

        zerpol = torch.zeros((Nx, Ny, int(radormax * azormax - 1), int(azormax) + 1)).to(self.dev)
        rhosq = x ** 2 + y ** 2
        rho = torch.sqrt(rhosq)
        zerpol[:, :, 0, 0] = torch.ones(x.size()).to(self.dev)

        for jm in range(int(azormax) + 1):
            m = jm * 1
            if m > 0:
                zerpol[:, :, jm, jm] = rho * zerpol[:, :, jm - 1, jm - 1]

            zerpol[:, :, jm + 2, jm] = ((m + 2) * rhosq - m - 1) * torch.squeeze(zerpol[:, :, jm, jm])
            itervalue = int(radormax) - 1 - m + 2

            for p in range(itervalue):
                piter = p + 2
                n = m + 2 * piter
                jn = n * 1

                zerpol[:, :, jn, jm] = (2 * (n - 1) * (n * (n - 2) * (2 * rhosq - 1) - m ** 2) * zerpol[:, :, jn - 2,
                                                                                                 jm] -
                                        n * (n + m - 2) * (n - m - 2) * zerpol[:, :, jn - 4, jm]) / (
                                               (n - 2) * (n + m) * (n - m))

        phi = torch.arctan2(y, x)

        allzernikes = torch.zeros((Nx, Ny, Nzer)).to(self.dev)

        for j in range(Nzer):
            n = int(orders[j, 0])
            m = int(orders[j, 1])
            if m >= 0:
                allzernikes[:, :, j] = zerpol[:, :, n, m] * torch.cos(m * phi)
            else:
                allzernikes[:, :, j] = zerpol[:, :, n, -m] * torch.sin(-m * phi)

        return allzernikes


    def get_pupil_matrix(self):

        PupilSize = 1.0
        DxyPupil = 2 * PupilSize / self.Npupil
        XYPupil = torch.arange(-PupilSize + DxyPupil / 2, PupilSize, DxyPupil).to(self.dev)
        YPupil, XPupil = torch.meshgrid(XYPupil, XYPupil, indexing='xy')

        argMed = 1 - (XPupil ** 2 + YPupil ** 2) * self.NA ** 2 / self.refmed ** 2
        phiMed = torch.arctan2(torch.tensor(0, dtype=torch.float).to(self.dev), argMed)
        complex1 = torch.tensor(1j, dtype=torch.complex64).to(self.dev)

        CosThetaMed = torch.sqrt(torch.abs(argMed)) * (torch.cos(phiMed / 2) - complex1 * torch.sin(phiMed / 2) - 0j)
        CosThetaCov = torch.sqrt(1 - (XPupil ** 2 + YPupil ** 2) * self.NA ** 2 / self.refcov ** 2 - 0j)
        CosThetaImm = torch.sqrt(1 - (XPupil ** 2 + YPupil ** 2) * self.NA ** 2 / self.refimm ** 2 - 0j)
        CosThetaImmnom = torch.sqrt(1 - (XPupil ** 2 + YPupil ** 2) * self.NA ** 2 / self.refimmnom ** 2 - 0j)

        FresnelPmedcov = 2 * self.refmed * CosThetaMed / (self.refmed * CosThetaCov + self.refcov * CosThetaMed)
        FresnelSmedcov = 2 * self.refmed * CosThetaMed / (self.refmed * CosThetaMed + self.refcov * CosThetaCov)
        FresnelPcovimm = 2 * self.refcov * CosThetaCov / (self.refcov * CosThetaImm + self.refimm * CosThetaCov)
        FresnelScovimm = 2 * self.refcov * CosThetaCov / (self.refcov * CosThetaCov + self.refimm * CosThetaImm)
        FresnelP = FresnelPmedcov * FresnelPcovimm
        FresnelS = FresnelSmedcov * FresnelScovimm

        Phi = torch.arctan2(YPupil, XPupil)
        CosPhi = torch.cos(Phi)
        SinPhi = torch.sin(Phi)
        CosTheta = CosThetaMed
        SinTheta = torch.sqrt(1 - CosTheta ** 2)

        pvec = torch.zeros((self.Npupil, self.Npupil, 3), dtype=torch.complex64).to(self.dev)
        svec = torch.zeros((self.Npupil, self.Npupil, 3), dtype=torch.complex64).to(self.dev)

        pvec[:, :, 0] = (FresnelP + 0j) * (CosTheta + 0j) * (CosPhi + 0j)
        pvec[:, :, 1] = FresnelP * CosTheta * (SinPhi + 0j)
        pvec[:, :, 2] = -FresnelP * (SinTheta + 0j)
        svec[:, :, 0] = -FresnelS * (SinPhi + 0j)
        svec[:, :, 1] = FresnelS * (CosPhi + 0j)
        svec[:, :, 2] = 0

        PolarizationVector = torch.zeros((self.Npupil, self.Npupil, 2, 3), dtype=torch.complex64).to(self.dev)
        PolarizationVector[:, :, 0, :] = CosPhi[:, :, None] * pvec - SinPhi[:, :, None] * svec
        PolarizationVector[:, :, 1, :] = SinPhi[:, :, None] * pvec + CosPhi[:, :, None] * svec

        ApertureMask = (XPupil ** 2 + YPupil ** 2) < 1.0

        Amplitude = ApertureMask * torch.sqrt(CosThetaImm) / (self.refmed * CosThetaMed)
        Amplitude[~ApertureMask] = 0 + 0j
        if self.zstack == True:
            #size_pup = XPupil.size()
            Waberration = torch.zeros((int(self.aberrations.size(0)),XPupil.size(0),XPupil.size(1)), dtype=torch.complex64).to(self.dev)
            orders = self.aberrations[0,:,0:2]
            zernikecoefs = self.aberrations[:,:, 2]
            normfac = torch.sqrt(2 * (orders[:, 0] + 1) / (1 + (orders[:, 1] == 0)))
            zernikecoefs = normfac * zernikecoefs
            all_zernikes = self.get_zernikefunctions(orders, XPupil, YPupil)
            Waberration = Waberration+ torch.sum(zernikecoefs[:, None,None,:] * all_zernikes[None, :, :, :], dim=-1)

            Waberration = Waberration + self.zvals[0] * self.refimm * CosThetaImm - self.zvals[
                1] * self.refimmnom * CosThetaImmnom - \
                          self.zvals[2] * self.refmed * CosThetaMed
            PhaseFactor = torch.exp(2 * torch.pi * complex1 * torch.real(Waberration / self.Lambda))
            Waberration = Waberration * ApertureMask[None,...]

            PupilMatrix = Amplitude[..., None, None] * PhaseFactor[..., None, None] * PolarizationVector
            PupilMatrix[:,~ApertureMask] = 0 + 0j
        else:
            size_pup = XPupil.size()
            Waberration = torch.zeros(size_pup, dtype=torch.complex64).to(self.dev)
            orders = self.aberrations[:, 0:2]
            zernikecoefs = self.aberrations[:, 2]
            normfac = torch.sqrt(2 * (orders[:, 0] + 1) / (1 + (orders[:, 1] == 0)))
            zernikecoefs = normfac * zernikecoefs
            all_zernikes = self.get_zernikefunctions(orders, XPupil, YPupil)

            for j in range(len(zernikecoefs)):
                Waberration = Waberration + zernikecoefs[j] * all_zernikes[:, :, j]

            Waberration = Waberration + self.zvals[0] * self.refimm * CosThetaImm - self.zvals[1] * self.refimmnom * CosThetaImmnom - \
                          self.zvals[2] * self.refmed * CosThetaMed
            PhaseFactor = torch.exp(2 * torch.pi * complex1 * torch.real(Waberration / self.Lambda))
            Waberration = Waberration * ApertureMask

            PupilMatrix = Amplitude[..., None, None] * PhaseFactor[..., None, None] * PolarizationVector
            PupilMatrix[~ApertureMask] = 0 + 0j

        wavevector = torch.zeros((XPupil.size()[0], XPupil.size()[0], 3), dtype=torch.complex64).to(self.dev)
        wavevector[:, :, 0] = (2 * torch.pi * self.NA / self.Lambda) * XPupil
        wavevector[:, :, 1] = (2 * torch.pi * self.NA / self.Lambda) * YPupil
        wavevector[:, :, 2] = (2 * torch.pi * self.refmed / self.Lambda) * CosThetaMed
        wavevectorzimm = (2 * torch.pi * self.refimm / self.Lambda) * CosThetaImm

        self.wavevector = wavevector
        self.wavevectorzimm = wavevectorzimm
        self.Waberration = Waberration
        self.all_zernikes = all_zernikes
        self.PupilMatrix = PupilMatrix



    def get_field_matrix_derivatives(self,theta):

        complex1 = torch.tensor(1j, dtype=torch.complex64).to(self.dev)
        dz = (abs(self.zmax) + abs(self.zmin)) / self.K

        Zimage = torch.linspace(self.zmin, self.zmax + dz, self.K).to(self.dev)
        numders = self.spatial_dims  # 3D



        num_beads = theta.size()[0]
        if self.zstack == True:

            orders = self.aberrations[:,:, 0:2]
            normfac = torch.sqrt(2 * (orders[:,:, 0] + 1) / (1 + (orders[:,:, 1] == 0)))
            xemit = theta[:,0]  # distance from center
            yemit = theta[:,1]
            zemit = theta[:,2]

            FieldMatrix = torch.zeros((theta.size(0),theta.size(1),self.Mx, self.My, self.K, 2, 3), dtype=torch.complex64).to(self.dev)
            FieldMatrixDerivatives = torch.zeros(
                (theta.size(0),theta.size(1),self.Mx, self.My, self.K, numders + self.aberrations.size()[1], 2, 3), dtype=torch.complex64
            ).to(self.dev)
            PupilFunctionDerivatives = torch.zeros(
                (theta.size(0),theta.size(1),self.Npupil, self.Npupil, 3, 2, 3), dtype=torch.complex64
            ).to(self.dev)  # pupil, pupil, dim, 2,3

            for jz in range(self.K):
                zemitrun = Zimage[jz]

                # phase contribution due to position of the emitter
                Wlateral = xemit[:,None,None] * self.wavevector[:, :, 0] + yemit[:,None,None] * self.wavevector[:, :, 1]
                Wpos = Wlateral + (zemit - zemitrun)[:,None,None] * self.wavevectorzimm[None,...]
                PositionPhaseMask = torch.exp(-complex1 * torch.real(Wpos))
                # Pupil function
                self.PupilFunction = PositionPhaseMask[..., None, None] * self.PupilMatrix

                # Pupil function for xy - derivatives
                PupilFunctionDerivatives[:,:,:, :, 0, :, :] = (-complex1 * self.wavevector[:, :, 0][
                    ..., None, None] * self.PupilFunction)[:,None,...]
                PupilFunctionDerivatives[:,:,:, :, 1, :, :] = (-complex1 * self.wavevector[:, :, 1][
                    ..., None, None] * self.PupilFunction)[:,None,...]

                # pupil functions for z-derivatives (only for xyz, stage)
                PupilFunctionDerivatives[:,:,:, :, 2, :, :]  = (-complex1 * self.wavevectorzimm[
                    ..., None, None] * self.PupilFunction)[:,None,...] # remove minus!!!


                IntermediateImage = self.cztfunc2D(self.PupilFunction, self.Ay, self.By, self.Dy)

                FieldMatrix[:,:,:, :, jz, :, :] = self.cztfunc2D(IntermediateImage, self.Ax, self.Bx, self.Dx)[:,None,...]
                IntermediateImage = self.cztfunc3D(PupilFunctionDerivatives[:,0,...], self.Ay, self.By, self.Dy)
                FieldMatrixDerivatives[:,:,:, :, jz, 0:numders, :, :] = self.cztfunc3D(IntermediateImage, self.Ax, self.Bx, self.Dx)[:,None,...]

                # for aberrations
                for jzer in range(self.aberrations.size()[1]):
                    jder = numders + jzer

                    self.PupilFunction = (
                                            2 * torch.pi * -complex1 * normfac[0,jzer] * self.all_zernikes[:, :,
                                                                                      jzer] / self.Lambda
                                    )[None,..., None, None] * (PositionPhaseMask[..., None, None] * self.PupilMatrix)
                    IntermediateImage = self.cztfunc2D(self.PupilFunction, self.Ay, self.By, self.Dy)
                    FieldMatrixDerivatives[:,:,:, :, jz, jder, :, :] = self.cztfunc2D(IntermediateImage, self.Ax, self.Bx, self.Dx)[:,None,...]

            # if True:
            #     import matplotlib.pyplot as plt
            #     import numpy as np
            #     jz = int(self.K / 2)
            #     numb = 0
            #     plt.figure()
            #     for itel in range(2):
            #         for jtel in range(3):
            #             tempim = FieldMatrix[:,:,jz,itel,jtel].detach().cpu().numpy()
            #             numb +=1
            #             plt.subplot(2,3,numb)
            #             plt.imshow(np.real(tempim))
            #             plt.title('real')
            #     plt.show()
            #
            #     numb = 0
            #     plt.figure()
            #     for itel in range(2):
            #         for jtel in range(3):
            #             tempim = FieldMatrix[:,:,jz,itel,jtel].detach().cpu().numpy()
            #             numb +=1
            #             plt.subplot(2,3,numb)
            #             plt.imshow(np.imag(tempim)*180/torch.pi)
            #             plt.title('imag')
            #     plt.show()
            return FieldMatrix[:,0,...], FieldMatrixDerivatives[:,0,...]

        else:
            orders = self.aberrations[:, 0:2]
            normfac = torch.sqrt(2 * (orders[:, 0] + 1) / (1 + (orders[:, 1] == 0)))
            xemit = theta[:, 0]  # distance from center
            yemit = theta[:, 1]
            zemit = theta[:, 2]
            wavevector = torch.tile(self.wavevector[None, ...], (num_beads, 1, 1, 1))
            Wlateral = xemit[..., None, None] * wavevector[:, :, :, 0] + yemit[..., None, None] * wavevector[:, :, :, 1]
            Wpos = Wlateral + (zemit[..., None, None]) * wavevector[:, :, :, 2]  # only for medium!
            PositionPhaseMask = torch.exp(-complex1 * torch.real(Wpos))
            # Pupil function
            self.PupilFunction = PositionPhaseMask[..., None, None] * self.PupilMatrix

            PupilFunctionDerivatives = torch.zeros(
                (num_beads, self.Npupil, self.Npupil, 3, 2, 3), dtype=torch.complex64
            ).to(self.dev)  # pupil, pupil, dim, 2,3
            # Pupil function for xy - derivatives
            PupilFunctionDerivatives[:, :, :, 0, :, :] = -complex1 * wavevector[:, :, :, 0][
                ..., None, None] * self.PupilFunction
            PupilFunctionDerivatives[:, :, :, 1, :, :] = -complex1 * wavevector[:, :, :, 1][
                ..., None, None] * self.PupilFunction

            # pupil functions for z-derivatives (only for xyz, stage)
            PupilFunctionDerivatives[:, :, :, 2, :, :] = (
                    -complex1 * wavevector[:, :, :, 2][..., None, None] * self.PupilFunction
            )# remove minus!!!


            IntermediateImage = self.cztfunc2D(self.PupilFunction, self.Ay, self.By, self.Dy)
            FieldMatrix = self.cztfunc2D(IntermediateImage, self.Ax, self.Bx, self.Dx)
            IntermediateImage = self.cztfunc3D(PupilFunctionDerivatives, self.Ay, self.By, self.Dy)
            FieldMatrixDerivatives = self.cztfunc3D(IntermediateImage, self.Ax, self.Bx, self.Dx)


            return FieldMatrix, FieldMatrixDerivatives
    def get_normalization(self):
        if self.zstack:
            # Intensity matrix
            IntensityMatrix = torch.zeros((self.PupilMatrix.size()[0], 3, 3))
            for itel in range(3):
                for jtel in range(3):
                    pupmat1 = self.PupilMatrix[:, :, :, :, itel]
                    pupmat2 = self.PupilMatrix[:, :, :, :, jtel]
                    IntensityMatrix[:, itel, jtel] = torch.sum(torch.real(pupmat1 * torch.conj(pupmat2)), dim=(1, 2, 3))

            # normalization to take into account discretization correctly
            DxyPupil = 2 * self.NA / self.Lambda / self.Npupil
            normfac = DxyPupil ** 2 / self.pixelsize ** 2
            IntensityMatrix = normfac * IntensityMatrix

            # evaluation normalization factors
            normint_free = torch.sum(torch.diagonal(IntensityMatrix, dim1=1, dim2=2), dim=1) / 3


        else:

            # Intensity matrix
            IntensityMatrix = torch.zeros((3, 3))
            for itel in range(3):
                for jtel in range(3):
                    pupmat1 = self.PupilMatrix[:, :, :, itel]
                    pupmat2 = self.PupilMatrix[:, :, :, jtel]
                    IntensityMatrix[itel, jtel] = torch.sum(torch.sum(torch.real(pupmat1 * torch.conj(pupmat2))))

            # normalization to take into account discretization correctly
            DxyPupil = 2 * self.NA / self.Lambda / self.Npupil
            normfac = DxyPupil ** 2 / self.pixelsize ** 2
            IntensityMatrix = normfac * IntensityMatrix

            # evaluation normalization factors
            normint_free = torch.sum(torch.diag(IntensityMatrix)) / 3

        return normint_free
    def get_psfs_derivatives(self):
        # PSF normalization (only freely diffusive)
        normint_free = (self.get_normalization()).to(self.dev)

        # if free and ZSTACK!
        if self.zstack:
            FreePSF = 1 / 3 * torch.sum(torch.sum(torch.abs(self.FieldMatrix) ** 2, dim=-1), dim=-1)
            tmp = torch.transpose(self.FieldMatrixDerivatives, 4, 5)
            tmpFieldMatrixDerivatives = torch.transpose(tmp, 5, 6)
            #tmpFieldMatrixDerivatives = torch.transpose(self.FieldMatrixDerivatives, [0,1,2,4,5,3])

            FreePSFder = (2 / 3) * torch.sum(torch.sum(torch.real(torch.conj(self.FieldMatrix[...,None]) *
                                                                  tmpFieldMatrixDerivatives), dim=-2), dim=-2)
            FreePSF = FreePSF / normint_free[...,None,None,None]
            FreePSFder = FreePSFder / normint_free[...,None,None,None,None]

        else:
            FreePSF = 1 / 3 * torch.sum(torch.sum(torch.abs(self.FieldMatrix) ** 2, dim=-1), dim=-1)
            tmp = torch.transpose(self.FieldMatrixDerivatives, 3, 4)
            tmpFieldMatrixDerivatives = torch.transpose(tmp, 4, 5)

            FreePSFder = (2 / 3) * torch.sum(torch.sum(torch.real(torch.conj(self.FieldMatrix[..., None]) *
                                                                  tmpFieldMatrixDerivatives), dim=-2), dim=-2)
            FreePSF = FreePSF / normint_free
            FreePSFder = FreePSFder / normint_free
        # TODO : no zstack



        # free
        PSF = FreePSF * 1
        PSFder = FreePSFder * 1

        return PSF, PSFder

    def poissonrate(self, theta):
        if self.zstack and self.aberrations.dim() == 2:
            self.aberrations = self.aberrations[None, ...].repeat([theta.size(0), 1, 1])
        if self.zstack or not hasattr(self, 'PupilMatrix'):
            self.get_pupil_matrix()


        self.FieldMatrix, self.FieldMatrixDerivatives = self.get_field_matrix_derivatives(theta)

        self.PSF, self.PSFder = self.get_psfs_derivatives()

        if self.zstack:
            mu = torch.zeros((theta.size(0), self.Mx, self.My, self.K)).to(self.dev)
            dmudtheta = torch.zeros((theta.size(0),self.Mx, self.My, self.K, self.numparams)).to(self.dev)
            mu[:,:, :, :] = theta[:,3][...,None,None,None] * self.PSF[:, :, :,:] + theta[:,4][...,None,None,None]
            dmudtheta[:,:, :, :, 0:3] = theta[:,3][...,None,None,None,None] * self.PSFder[:,:, :, :, 0:3]#+ theta[:,4][...,None,None,None,None]
            dmudtheta[:,:, :, :, 3] = self.PSF
            dmudtheta[:,:, :, :, 4] = 1
            dmudtheta[:,:, :, :, 5::] = -theta[:,3][...,None,None,None,None] * self.PSFder[:,:, :, :, 3::]


        else:
            numbeads = theta.size()[0]
            mu = torch.zeros((numbeads, self.Mx, self.My)).to(self.dev)

            dmudtheta = torch.zeros((numbeads, self.Mx, self.My, 2+self.spatial_dims)).to(self.dev)
            mu[:, :, :] = theta[:, 3, None, None] * self.PSF[:, :, :] + theta[:, 4, None, None]
            dmudtheta[:, :, :, 0:3] = theta[:, 3, None, None, None] * self.PSFder[:, :, :, 0:3]
            dmudtheta[:, :, :, 3] = self.PSF
            dmudtheta[:, :, :, 4] = 1
            # dmudtheta[:, :, :, 5::] = theta[3] * self.PSFder[:, :, 3::]

        return mu, dmudtheta

    def cztfunc2D(self, datain, Amt, Bmt, Dmt):

        if self.zstack:
            cztin = torch.zeros((datain.size()[0], datain.size()[1], self.L, 2, 3), dtype=torch.complex64).to(self.dev)

            cztin[:, :, 0:self.N, :, :] = Amt[None,..., None, None] * datain
            temp = Dmt[None,..., None, None] * torch.fft.fft(cztin, dim=2)
            cztout = torch.fft.ifft(temp, dim=2)
            dataout = Bmt[None,..., None, None] * cztout[:, :, 0:self.M, :, :]
            dataout = torch.transpose(dataout, 1, 2)

        else:
            cztin = torch.zeros((datain.size()[0], datain.size()[1], self.L, 2, 3), dtype=torch.complex64).to(self.dev)
            cztin[:, :, 0:self.N, :, :] = Amt[None, ..., None, None] * datain
            temp = Dmt[None, ..., None, None] * torch.fft.fft(cztin, dim=2)
            cztout = torch.fft.ifft(temp, dim=2)
            dataout = Bmt[None, ..., None, None] * cztout[:, :, 0:self.M, :, :]
            dataout = torch.transpose(dataout, 1, 2)
        return dataout

    def cztfunc3D(self, datain, Amt, Bmt, Dmt):
        if self.zstack:
            dim = 3  # 3 dimensions
            cztin = torch.zeros((datain.size()[0], datain.size()[1], self.L, dim, 2, 3), dtype=torch.complex64).to(
                self.dev)

            cztin[:, :, 0:self.N, :, :, :] = Amt[None,..., None, None, None] * datain
            temp = Dmt[..., None, None, None] * torch.fft.fft(cztin, dim=2)
            cztout = torch.fft.ifft(temp, dim=2)
            dataout = Bmt[..., None, None, None] * cztout[:, :, 0:self.M, :, :, :]
            dataout = torch.transpose(dataout, 1, 2)


        else:
            dim = 3  # 3 dimensions
            cztin = torch.zeros((datain.size()[0], datain.size()[1], self.L, dim, 2, 3), dtype=torch.complex64).to(self.dev)

            cztin[:, :, 0:self.N, :, :, :] = Amt[None, ..., None, None, None] * datain
            temp = Dmt[None, ..., None, None, None] * torch.fft.fft(cztin, dim=2)
            cztout = torch.fft.ifft(temp, dim=2)
            dataout = Bmt[None, ..., None, None, None] * cztout[:, :, 0:self.M, :, :, :]
            dataout = torch.transpose(dataout, 1, 2)
        return dataout

    def compute_crlb(self,mu, jac):
        """
        Compute crlb from expected value and per pixel derivatives.
        mu: [N, H, W]
        jac: [N, H,W, coords]
        """

        naxes = jac.shape[-1]
        axes = [i for i in range(naxes)]
        jac = jac[..., axes]

        sample_dims = tuple(torch.arange(1, len(mu.shape)))
        fisher = torch.matmul(jac[..., None], jac[..., None, :])  # derivative contribution
        fisher = fisher / mu[..., None, None]  # px value contribution
        fisher = fisher.sum(sample_dims)

        crlb = torch.zeros((len(mu), naxes), device=mu.device)
        crlb[:, axes] = torch.sqrt(torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2))
        return crlb
    def likelihood(self,image, mu, dmudtheta):
        numparams = dmudtheta.shape[-1]

        keps = 1e3 * torch.finfo(mu.dtype).eps
        mupos = torch.where(mu > 0, mu, keps)
        varfit = 0
        weight = (image - mupos) / (mupos + varfit)
        dweight = (image + varfit) / (mupos + varfit) ** 2


        if self.zstack:
            sampledim = (1,2,3)
        else:
            sampledim = (1,2)
        logL = torch.sum((image + varfit) * torch.log(mupos + varfit) - (mupos + varfit), dim=sampledim)
        gradlogL = torch.sum(weight[..., None] * dmudtheta, dim=sampledim)
        # if self.zstack:
        #     HessianlogL = torch.zeros((numparams, numparams))
        #
        #     for ii in range(numparams):
        #         for jj in range(numparams):
        #             HessianlogL[ii, jj] = torch.sum(-dweight * dmudtheta[..., ii] * dmudtheta[..., jj])
        # else:
        HessianlogL = torch.zeros((gradlogL.size(0), numparams, numparams))

        for ii in range(numparams):
            for jj in range(numparams):
                HessianlogL[:, ii, jj] = torch.sum(-dweight * dmudtheta[..., ii] * dmudtheta[..., jj], sampledim)

        return logL, gradlogL, HessianlogL
    def fit_zstack(self, data_tot):
        def thetalimits(aberrations, Lambda, Mx, My, pixelsize, zspread, dev, mu_all, zstack=False):
            if self.zstack and self.aberrations.dim() == 2:
                aberrations = aberrations[None, ...].repeat([mu_all.size(0), 1, 1])
            zernikecoefsmax = 0.25 * Lambda * torch.ones((aberrations.size()[1])).to(dev)
            roisizex = Mx * pixelsize
            roisizey = My * pixelsize
            xmin = -roisizex / 2.4
            xmax = roisizex / 2.4
            ymin = -roisizey / 2.4
            ymax = roisizey / 2.4
            zmin = -1000
            zmax = 1000

            if zstack:
                thetamin = torch.concat((torch.tensor([xmin, ymin, zmin, 1, 0]).to(dev), -zernikecoefsmax), dim=0)
                thetamax = torch.concat((torch.tensor([xmax, ymax, zmax, 1e6, 1e5]).to(dev), zernikecoefsmax), dim=0)
            else:
                thetamin = torch.tensor([xmin, ymin, zmin, 1, 0]).to(dev)
                thetamax = torch.tensor([xmax, ymax, zmax, 1e6, 1e5]).to(dev)

            return thetamin, thetamax



        mu_all = data_tot * 1

        # Check dimensions of `mu_all`
        if mu_all.dim() == 3:
            mu_all = mu_all.unsqueeze(0)
        self.K = mu_all.size(1)




        theta_zstack_all = self.init_guess_based_on_zeroaberrations_forzstack(mu_all)
        theta_zstack = torch.zeros((mu_all.size(0),self.numparams)).to(self.dev)
        theta_zstack[:, 0:self.spatial_dims + 2] = theta_zstack_all[:, :]
        #theta_zstack[:,3] = 2000
        #theta_zstack[:,4] = 10
        theta_zstack[:,5::] = self.aberrations[:,2]
        zmin = self.zrange[0]
        zmax = self.zrange[1]

       # theta_zstack[:,self.spatial_dims + 2::] = self.aberrations[:,:, 2]


        Kz = self.K

        thetamin, thetamax = thetalimits(self.aberrations, self.Lambda, self.Mx, self.My,self.pixelsize, self.zspread, self.dev,mu_all, zstack=True)

        # thetaretry = (thetamax + thetamin) / 2
        # mu, dmudtheta = self.poissonrate(theta_zstack)
        paramrange = torch.concatenate((thetamin[...,None],thetamax[...,None]),dim=1)
        mu_all = torch.permute(mu_all,(0,2,3,1))
        theta, traces = self.LM_MLE_for_zstack(mu_all,paramrange,theta_zstack)

        finalmu,_ = self.poissonrate(theta)
        return finalmu, theta, traces
            #thetastore[i, :] = theta
            #
            # mu = np.transpose(mu, [2, 0, 1])
            # spots = np.transpose(spots, [2, 0, 1])
            #
            # mu_all[i, :, :, :] = mu * 1

    def init_guess_based_on_zeroaberrations_forzstack(self, beadstack,photons=0, bg=0):
        # beadstack has size of [batch,zlices,roi,roi]
        # aberrations_original = self.aberrations*1 # store aberrations
        # zstack_original = copy.copy(self.zstack)
        # itersoriginal = copy.copy(self.Nitermax)
        # self.zstack = False
        # self.aberrations[:,2] = 0
        # self.Nitermax= 10
        # if photons == 0:
        bg = torch.median(beadstack[:, :, :])
        photons = torch.mean(torch.sum(beadstack,dim=(-1,-2))) - (bg*self.Mx**2)
        # photons = ((torch.max(torch.max(beadstack[:, :, :])) - bg) * (
        #         2 * torch.pi * self.NA * (
        #         self.Lambda / (4 * self.NA)) ** 2) ** 0.5)/self.pixelsize  # initial guess, might need to be better
        # if bg == 0:


        #change this to estimate photons, position and background

        middle_plane = int(self.K/2)
        beadstack_singleplane = beadstack[:,middle_plane,:,:]
        init_guess = torch.tensor([0,0,0,photons,bg]).to(self.dev)
        init_guess = init_guess[None,...].repeat([beadstack_singleplane.size(0),1])

        # param_range = torch.tensor([
        #     [-(self.Mx / 2 - 2) * self.pixelsize, (self.Mx / 2 - 2) * self.pixelsize],
        #     [-(self.Mx / 2 - 2) * self.pixelsize, (self.Mx / 2 - 2) * self.pixelsize],
        #     [0, 0],
        #     [1, 1e9],
        #     [0.5, 1000],
        # ])
        param_range = torch.tensor([
            [0,0],
            [0,0],
            [0, 0],
            [1, 1e9],
            [0.5, 1000],
        ])


       # initguess,_ = self.LM_MLE(beadstack_singleplane, param_range,init_guess)
       #
       #  self.aberrations= aberrations_original
       #  self.zstack = zstack_original
       #  self.Nitermax = itersoriginal
        return init_guess

    def LM_MLE(self, smp, min_max,initial_guess):

        cur = initial_guess * 1
        mu = torch.zeros(smp.size()).to(self.dev)
        jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2], cur.size()[1])).to(self.dev)


        traces = torch.zeros((self.Nitermax + 1, cur.size()[0], cur.size()[1])).to(self.dev)
        traces[0, :, :] = cur

        tol = torch.ones((cur.size()[0], cur.size()[1])).to(self.dev) * self.tollim
        good_array = torch.ones(cur.size()[0]).to(self.dev).type(torch.bool)
        delta = torch.ones(cur.size()).to(self.dev)
        bool_array = torch.ones(cur.size()).to(self.dev).type(torch.bool)

        i = 0
        flag_tolerance = 0
        loglik_old = 10000
        damping_original = self.damping_lm*1
        while (i < self.Nitermax) and (flag_tolerance == 0):

            mu[good_array, :, :], jac[good_array, :, :, :] = self.poissonrate(cur[good_array, :])

            #jac[good_array, :, :, 2] *=1e3

            cur[good_array, :],loglik  = self.MLE_update(cur[good_array, :],mu[good_array, :, :],jac[good_array, :, :, :],
                                             smp[good_array, :, :], min_max)


            traces[i + 1, good_array, :] = cur[good_array, :]
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :]-traces[i, good_array, :])/traces[i, good_array, :]

            bool_array[good_array] = (delta[good_array, :] < tol[good_array, :]).type(torch.bool)
            test = torch.sum(bool_array, dim=1)
            good_array = test != self.spatial_dims+2 # two for photons and bg


            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
            loglik_old = loglik*1
        return cur, traces



    def MLE_update(self,cur, mu, jac, smp, param_range_min_max):
        """
        Separate some of the calculations to speed up with jit script
        """

        merit, grad, Hessian = self.likelihood(smp, mu, jac)
        diag = torch.diagonal(Hessian, dim1=-2, dim2=-1)
        b = torch.eye(diag.size(1))
        c = diag.unsqueeze(2).expand(diag.size(0), diag.size(1), diag.size(1))
        diag_full = c * b
        # matty = Hessian + lambda_ * diag_full

        # thetaupdate
        # update of fit parameters via Levenberg-Marquardt
        Bmat = Hessian + self.damping_lm * diag_full
        Bmat = Bmat.to(device=self.dev)

        dtheta = torch.linalg.solve(-Bmat, grad)

        dtheta[torch.isnan(dtheta)] = -0.1 * cur[torch.isnan(dtheta)]
        cur = cur + dtheta

        cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
        cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))


        return cur, merit

    def LM_MLE_for_zstack(self, smp, min_max, initial_guess):
        if self.zstack and self.aberrations.dim() == 2:
            self.aberrations = self.aberrations[None, ...].repeat([initial_guess.size(0), 1, 1])
        cur = initial_guess * 1
        mu = torch.zeros(smp.size()).to(self.dev)
        if self.zstack:
            jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2],smp.size()[3], cur.size()[1])).to(self.dev)
        else:

            jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2], cur.size()[1])).to(self.dev)

        traces = torch.zeros((self.Nitermax + 1, cur.size()[0], cur.size()[1])).to(self.dev)
        traces[0, :, :] = cur

        tol = torch.ones((cur.size()[0], cur.size()[1])).to(self.dev) * self.tollim
        good_array = torch.ones(cur.size()[0]).to(self.dev).type(torch.bool)
        delta = torch.ones(cur.size()).to(self.dev)
        bool_array = torch.ones(cur.size()).to(self.dev).type(torch.bool)

        i = 0
        flag_tolerance = 0
        while (i < self.Nitermax) and (flag_tolerance == 0):

            mu[good_array, ...], jac[good_array, ...] = self.poissonrate(cur[good_array, :])

            cur[good_array, ...] = self.MLE_update_forzstack(cur[good_array, ...], mu[good_array, ...], jac[good_array, ...],
                                                 smp[good_array, ...], min_max)

            traces[i + 1, good_array, ...] = cur[good_array, ...]
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :]) / traces[i,
                                                                                                             good_array,
                                                                                                             :]

            bool_array[good_array] = (delta[good_array, ...] < tol[good_array, ...]).type(torch.bool)
            test = torch.sum(bool_array, dim=1)
            good_array = test != self.spatial_dims + 2  # two for photons and bg
            self.aberrations[:,:,2] = cur[:, self.spatial_dims + 2::]
            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
        return cur, traces

    def torch_to_numpy(self,tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    def MLE_update_forzstack(self,cur, mu, jac, smp, param_range_min_max):
        """
        Separate some of the calculations to speed up with jit script
        """
        if self.zstack:
            sample_dim = (1,2,3,4)
        else:
            sample_dim = (1, 2, 3)
        merit, grad, Hessian = self.likelihood(smp, mu, jac)

        diag = torch.diagonal(Hessian, dim1=-2, dim2=-1)
        b = torch.eye(diag.size(1))
        c = diag.unsqueeze(2).expand(diag.size(0), diag.size(1), diag.size(1))
        diag_full = c * b
        # matty = Hessian + lambda_ * diag_full

        # thetaupdate
        # update of fit parameters via Levenberg-Marquardt
        Bmat = Hessian + self.damping_lm * diag_full
        Bmat = Bmat.to(device=self.dev)

        dtheta = torch.linalg.solve(-Bmat, grad)

        dtheta[torch.isnan(dtheta)] = -0.1 * cur[torch.isnan(dtheta)]
        cur = cur + dtheta

        cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
        cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))


        return cur

    def fit_emitters_batched(self, smp, roipos, savefn = 'test'):


        param_range = torch.tensor([
            [-(self.Mx / 2 - 2) * self.pixelsize, (self.Mx / 2 - 2) * self.pixelsize],
            [-(self.Mx / 2 - 2) * self.pixelsize, (self.Mx / 2 - 2) * self.pixelsize],
            [self.zspread[0], self.zspread[1]],
            [1, 1e9],
            [0.5, 1000],
        ])
        
        def find_first_zero_indices(tensor):
            zero_indices = (tensor == 0).nonzero()
            result = torch.zeros(tensor.size(1), dtype=torch.int64)

            for col in range(tensor.size(1)):
                col_indices = zero_indices[zero_indices[:, 1] == col]
                if col_indices.numel() > 0:
                    result[col] = col_indices[0, 0]
                else:
                    result[col] = tensor.size(0) - 1

            return result

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

        estim_list = []
        iterations_vector_tot = []
        crlb_tot = []
        mu_list = []

        numspots = smp.size(0)
        dx = (0 * torch.rand((numspots, 1))).to(self.dev) * self.pixelsize
        dy = (0 * torch.rand((numspots, 1))).to(self.dev) * self.pixelsize
        dz = (0 * torch.rand((numspots, 1))).to(self.dev) * self.pixelsize
        Nbackground = torch.mean(smp[:,0,:], dim=-1, keepdim=True).to(self.dev)
        Nphotons = ((torch.sum(smp, dim=(-1, -2)))[...,None] - (Nbackground * self.Mx**2).to(self.dev))
        initial_guess = torch.concat((dx, dy, dz, Nphotons, Nbackground), axis=1).to(self.dev)

        
        # Print the chunked tensors
        chunked_spots = partition_tensor(smp, self.batch_size)
        chunked_init_guess = partition_tensor(initial_guess, self.batch_size)
        for idx, smp in enumerate(chunked_spots):
            estim, traces = self.LM_MLE(smp, param_range, chunked_init_guess[idx])
            mu, jac = self.poissonrate(estim)

            crlb = self.compute_crlb(mu, jac)
            estim_list.append(estim)
            iterations_vector = find_first_zero_indices(traces[:, :, -1])
            iterations_vector_tot.append(iterations_vector)
            crlb_tot.append(crlb)
            mu_list.append(mu)
        traces_np = torch.permute(traces,(1,0,2)).detach().cpu().numpy()
        estim_final = torch.cat(estim_list)
        estim_final[:, 0:2] = estim_final[:, 0:2] / self.pixelsize + self.Mx / 2
        mu_final = torch.cat(mu_list)
        iterations_final = torch.cat(iterations_vector_tot).to(self.dev)
        crlb_final = torch.cat(crlb_tot)
        border = 2.1
        sel_pos = ((estim_final[:, 0] > border) & (estim_final[:, 0] < self.Mx - border - 1) &
                   (estim_final[:, 1] > border) & (estim_final[:, 1] < self.Mx - border - 1) &
                   (estim_final[:, 2] > self.zspread[0]) & (estim_final[:, 2] < self.zspread[1]))

        sel = torch.logical_and(sel_pos, (iterations_final < self.Nitermax))

        print(
          )

        roi_index = torch.arange(0,smp.size(0)).to(self.dev)
        self.roi_pos_filtered = roipos[sel,...]
        self.roi_index_filtered = roi_index[sel]
        self.estim_filtered = estim_final[sel,...]
        mu_filtered = mu_final[sel,...]
        smp_filtered = smp[sel,...]
        self.crlb_filtered = crlb_final[sel,...]
        self.saveHDF5(savefn)

        # Generate random indices
        random_indices = torch.randperm(mu_filtered.size(0))[:min(1000,int(mu_filtered.size(0)))]
        s1 =  f'Filtering on position in ROI: {estim_final.size(0) - sel_pos.sum()}/{estim_final.size(0)} spots removed.\n'+\
              f'Filtering on iterations : {estim_final.size(0) - (iterations_final < self.Nitermax).sum()}/{estim_final.size(0)} spots removed.\n'

        return self.torch_to_numpy(mu_filtered[random_indices,...]), self.torch_to_numpy(smp_filtered[random_indices]), s1
    def saveHDF5(self, fn, saveGroups=False, fields=None):
        print(f"Saving hdf5 to {fn}")
        import h5py
        import yaml
        import os
        if fields is None:
            fields = []

        with h5py.File(fn, 'w') as f:
            dtype = [('frame', '<u4'),
                     ('x', '<f4'), ('y', '<f4'),
                     ('photons', '<f4'),
                     ('sx', '<f4'), ('sy', '<f4'),
                     ('bg', '<f4'),
                     ('lpx', '<f4'), ('lpy', '<f4'),
                     ('lI', '<f4'), ('lbg', '<f4'),
                     ('ellipticity', '<f4'),
                     ('net_gradient', '<f4'),
                     ('roi_index', '<i4'),
                     ('chisq', '<f4')]

            if saveGroups:
                dtype.append(('group', '<u4'))



            for fld in fields:
                dtype.append((fld, self.data.dtype[fld]))


            for fld in [('z', '<f4'), ('lpz', '<f4')]:
                dtype.append(fld)

            locs = f.create_dataset('locs', shape=(len(self.roi_index_filtered),), dtype=dtype)
            locs['frame'] = self.torch_to_numpy(self.roi_pos_filtered[:,0])
            locs['x'] = self.torch_to_numpy(self.estim_filtered[:, 0] + self.roi_pos_filtered[:,2])
            locs['y'] = self.torch_to_numpy(self.estim_filtered[:, 1] + self.roi_pos_filtered[:,1])
            locs['lpx'] = self.torch_to_numpy(self.crlb_filtered[:, 0])/self.pixelsize
            locs['lpy'] = self.torch_to_numpy(self.crlb_filtered[:, 1])/self.pixelsize

            locs['z'] = self.torch_to_numpy(self.estim_filtered[:, 2])
            locs['lpz'] = self.torch_to_numpy(self.crlb_filtered[:, 2])/self.pixelsize

            locs['photons'] = self.torch_to_numpy(self.estim_filtered[:, 3])
            locs['bg'] = self.torch_to_numpy(self.estim_filtered[:, 4])
            locs['lI'] = self.torch_to_numpy(self.crlb_filtered[:, 3])
            locs['lbg'] = self.torch_to_numpy(self.crlb_filtered[:, 4])
            locs['net_gradient'] = 0

            locs['roi_index'] = self.torch_to_numpy(self.roi_index_filtered)  # index into original un-filtered list of detected ROIs



            info = {'Byte Order': '<',
                    'Camera': 'Dont know',
                    'Data Type': 'uint16',
                    'File': fn,
                    'Frames': int(torch.max(self.roi_pos_filtered[:,0])),
                    'Width': int(self.imgwidth),
                    'Height': int(self.imgheight)
                    }

            info_fn = os.path.splitext(fn)[0] + ".yaml"
            with open(info_fn, "w") as file:
                yaml.dump(info, file, default_flow_style=False)

            f.close()
