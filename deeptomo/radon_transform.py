import numpy as np
import torch
from torch import nn
import torchkbnufft as tkbn
import einops
import skimage
import matplotlib.pyplot as plt


def kspace_builder(Nphi, Nrad, kmax):
    """
    Builds a kspace trajectory.
    The number of radial points needs to be odd (for now)

    Parameters
    ----------
    Nphi : Number of angles
    Nrad : Number of radial points
    kmax : Maximum value to be sampled in k-space

    Returns
    -------
    ktraj: a torch array of our kspace trajectory

    """
    assert Nrad%2==1
    phis = np.linspace(0, np.pi, Nphi)
    spoke = np.linspace(-kmax, kmax, Nrad, True)
    kx = []
    ky = []
    for phi in phis:
        for rad in spoke:
            tkx = np.sin(phi) * rad
            tky = np.cos(phi) * rad
            kx.append(tkx)
            ky.append(tky)
    result = torch.Tensor(np.stack((kx, ky), axis=0))
    return result


class radon_forward(nn.Module):
    """
    This class provides easy access to the forward radon transform
    """
    def __init__(self, image_shape, sino_shape, kspace, device):
        """
        Initialize radon transform object

        Parameters
        ----------
        image_shape : The shape of the real space image
        sino_shape : The shape of the sinogram
        kspace : The kspace trajectory
        """
        super().__init__()

        self.image_shape = image_shape
        self.sino_shape = sino_shape
        self.kspace = kspace
        self.device = device
        self.norm="ortho"
        self.nufft_object = tkbn.KbNufft(im_size=self.image_shape).to(self.device)

    def forward(self, image):
        """
        Compute the radon transform of an image

        Parameters
        ----------
        image : An image

        Returns
        -------
        kdata: the radon transform of that image

        """
        # first we do a NUFFT
        kdata = self.nufft_object(image.to(torch.complex64).to(self.device), self.kspace, norm=self.norm)
        kdata = einops.rearrange(kdata, "() () (N M) -> N M",
                                 N=self.sino_shape[0],
                                 M=self.sino_shape[1])
        # shift
        kdata = torch.fft.ifftshift(kdata, dim=1)
        kdata = torch.fft.ifft(kdata, dim=1).real
        kdata = torch.fft.fftshift(kdata, dim=1)
        return kdata

class radon_backward(nn.Module):
    def __init__(self, image_shape, sino_shape, kspace, device):
        """
        Initialize an inverse radon transform

        Parameters
        ----------
        image_shape : The shape of the real space image
        sino_shape : The shape of the sinogram image
        kspace : The kspace trajectory
        """

        super().__init__()

        self.image_shape = image_shape
        self.sino_shape = sino_shape
        self.kspace = kspace
        self.device = device

        self.norm = "ortho"  # ensure a consistent normalization
        self.dcomp = tkbn.calc_density_compensation_function(ktraj=self.kspace,
                                                             im_size=self.image_shape).to(self.device)
        self.adjoint_nufft_object = tkbn.KbNufftAdjoint(im_size=self.image_shape).to(self.device)

    def forward(self, sinogram):
        """
        Do a filtered backprojection given the sinogram

        Parameters
        ----------
        sinogram : The sinogram

        Returns
        -------
        An image
        """

        # We first do an fft
        kdata = torch.fft.ifftshift(sinogram.to(torch.complex64), axis=1)
        kdata = torch.fft.fft(kdata,axis=1)
        kdata = torch.fft.fftshift(kdata, axis=1)
        # now we do an adjoint kbnufft
        kdata = einops.rearrange(kdata, "N M -> () () (N M) ")
        result = self.adjoint_nufft_object(kdata*self.dcomp,
                                           self.kspace,
                                           norm=self.norm).real
        return result


def test_forward_backward():
    data = skimage.data.brain()[9].astype(float)
    image_shape = data.shape
    tensor_data = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    device = "cuda"


    Nphi = 180
    Nrad = 251
    sino_shape = (Nphi, Nrad)
    ktraj = kspace_builder(Nphi,Nrad, 2.0).to(device)



    rf = radon_forward(image_shape, sino_shape, ktraj, device).to(device)
    reci = rf(tensor_data.to(device)).cpu()
    plt.imshow(reci.numpy().real)
    plt.colorbar()
    plt.show()

    rb = radon_backward(image_shape, sino_shape, ktraj, device).to(device)
    tmp = rb(reci.to(device)).cpu()
    plt.imshow(tmp.numpy()[0,0,...].real)
    plt.show()

    recon_sino =rf(tmp.to(device)).cpu().numpy()
    a = reci.numpy().real
    norma = np.mean(reci.numpy().real)
    b = recon_sino.real
    normb = np.mean(recon_sino.real)
    a = a / norma
    b = b / normb
    plt.imshow(a)
    plt.colorbar()
    plt.show()
    plt.imshow(b)
    plt.colorbar()
    plt.show()

    print(norma,normb,norma/normb, normb/norma)
    plt.imshow(a-b)
    plt.colorbar()
    plt.show()




if __name__ == "__main__":
    test_forward_backward()









