import numpy as np
import pandas as pd
import astropy.units as u
from scipy.optimize import curve_fit

LAMBDA_LASER = 632.816 * u.nm

def convert(lam_spectre):
	return (LAMBDA_LASER**(-1) - lam_spectre**(-1)).to(u.cm**(-1))

def gaussian(x, mu, sigma):
	return 1 / sigma / np.sqrt(2) * np.exp(-(x-mu)**2/ sigma**2 / 2)

fente_500 = pd.read_csv("data/raman/fente_500.txt",
				names = ["lam", "cps"], sep="\t", skiprows=1)
fente_200 = pd.read_csv("data/raman/fente_200.txt",
				names = ["lam", "cps"], sep="\t", skiprows=1)
fente_50 = pd.read_csv("data/raman/fente_50.txt",
				names = ["lam", "cps"], sep="\t", skiprows=1)

if __name__ == "__main__":
	popt, pcov = curve_fit(gaussian, 
		xdata=fente_50.values[:, 0],
		ydata=fente_50.values[:, 1])
	print(popt)
