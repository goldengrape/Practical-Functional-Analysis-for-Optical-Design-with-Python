import numpy as np

class SimpleOpticalOperators:
    def __init__(self, size=32):
        self.size = size
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        self.X, self.Y = np.meshgrid(x, y)
    
    def fourier_transform(self, field):
        from scipy.fft import fft2, fftshift
        return fftshift(fft2(field))
    
    def lens_operator(self, field, focal_length):
        k = 2 * np.pi / 500e-9
        phase = np.exp(-1j * k * (self.X**2 + self.Y**2) / (2 * focal_length))
        return field * phase

if __name__ == "__main__":
    optics = SimpleOpticalOperators()
    field = np.exp(-(optics.X**2 + optics.Y**2))
    spectrum = optics.fourier_transform(field)
    print("Simple optical operators working!")