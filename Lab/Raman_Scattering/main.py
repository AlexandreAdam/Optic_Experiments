import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import c, k_B, hbar
from numpy import pi
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
}

matplotlib.rcParams.update(params)

LAMBDA_LASER = 6328.16 # Angstrom
lam_laser = 6328.16 * u.Angstrom
f_spectro = 220 * u.mm
n = 1200 / u.mm
results = {}
TEXFILE = "report/raman_scattering.tex"

def format_tex(n, prec:int=3) -> str:
    s = "{:.{prec}e}".format(n, prec=prec)
    s = re.sub(r'([0-9]\.[0-9]+)(e)(\+*)(-*)(0*)([0-9]+)', r'\g<1> \\times 10^{\g<4>\g<6>}', s)
    return s

def write_to_tex():
    with open(TEXFILE, "r", encoding='utf-8') as file:
        content = file.read()
    for k,v in results.items():
        content = re.sub(r'(pyoutput)(\{%s\})(\{\$*[\\A-Za-z0-9.,^\s-]+[\{0-9\}-]*\$*\})' % (k), r'\g<1>\g<2>{%s}' % (v), content) 
    with open(TEXFILE, "w", encoding='utf-8') as file:
        file.write(content)

def convert(lam_spectre):
	return (LAMBDA_LASER**(-1) - lam_spectre**(-1)) * 10**8 # cm-1

def gaussian(x, mu, sigma, A):
	return A * np.exp(-(x-mu)**2/ sigma**2 / 2)

def invert(lam_raman):
    return (LAMBDA_LASER**(-1) - 10**(-8) * lam_raman)**(-1)

def FWHM(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma

def omega(lam):
	return 2 * pi * c / lam

def temperature(Ias, Is, lam):
	# lam en cm^{-1}
	lam = 1 / lam 
	return -hbar * omega(lam) / k_B * (np.log(Ias / Is * ((omega(lam_laser) - omega(lam))/(omega(lam_laser) + omega(lam)))**4))**(-1)

def fentes():
	global results

	fente_150 = pd.read_csv("data/2019-10-02/fente-150",
					names = ["lam", "cps"], sep="\t", skiprows=1)
	fente_500 = pd.read_csv("data/raman/fente_500.txt",
					names = ["lam", "cps"], sep="\t", skiprows=1)
	fente_200 = pd.read_csv("data/2019-10-02/fente-200",
					names = ["lam", "cps"], sep="\t", skiprows=1)

	fente_150["lam"] = fente_150["lam"] / 10 # nm
	fente_200["lam"] = fente_200["lam"] / 10
	fente_500["lam"] = fente_500["lam"] / 10

	mask = (fente_500["lam"] > 691) & (fente_500["lam"] < 694.5)
	popt500, pcov500 = curve_fit(gaussian, 
				                xdata=fente_500["lam"][mask],
				                ydata=fente_500["cps"][mask],
				                p0=[693.2, 0.3, 730])
	mask = (fente_200["lam"] > 691) & (fente_200["lam"] < 694.5)
	popt200, pcov200 = curve_fit(gaussian, 
				                xdata=fente_200["lam"][mask],
				                ydata=fente_200["cps"][mask],
				                p0=[693.2, 0.3, 420])
	mask = (fente_150["lam"] > 691) & (fente_150["lam"] < 694.5)
	popt150, pcov150 = curve_fit(gaussian, 
				                xdata=fente_150["lam"][mask],
				                ydata=fente_150["cps"][mask],
				                p0=[693.2, 0.3, 730])
	print(popt150)
	print(popt200)
	print(popt500)
	results.update({"fente150": fr"${(FWHM(popt150[1]) * u.nm / 150 / u.micron).to(u.nm / u.mm).value:.2f} \\pm {(np.sqrt(pcov150[1, 1]) * u.nm / 150 / u.micron).to(u.nm / u.mm).value:.2f}$"})
	results.update({"fente200": fr"${(FWHM(popt200[1]) * u.nm / 200 / u.micron).to(u.nm / u.mm).value:.1f} \\pm {(np.sqrt(pcov200[1, 1]) * u.nm / 200 / u.micron).to(u.nm / u.mm).value:.1f}$"})
	results.update({"fente500": fr"${(FWHM(popt500[1]) * u.nm / 500 / u.micron).to(u.nm / u.mm).value:.2f} \\pm {(np.sqrt(pcov500[1, 1]) * u.nm / 500 / u.micron).to(u.nm / u.mm).value:.2f}$"})
	results.update({"fenteTH":  fr"{10**6 / 2 / n / f_spectro:.2f}"})
	print(results)

def echantillon1():
	e1 = pd.read_csv("data/2019-10-02/e1-100-6600-6700",
				names = ["lam", "cps"], sep="\t", skiprows=1)
	e1["lam"] = e1["lam"].apply(convert)
	mask = (e1.lam > 750)&(e1.lam < 800)
	plt.style.use('classic')
	fig, ax = plt.subplots()
	ax.plot(e1[mask].lam, e1[mask].cps, color="k", label="Données")

	plt.axvline(777, color="r", label="4H-SiC", alpha=0.8, linewidth=2, ls="--")
	plt.axvline(788, color="forestgreen", label="6H-SiC", alpha=0.8, linewidth=2, ls="--")
	#plt.axvline(767.5, color="forestgreen", alpha=0.8, linewidth=2, ls="--")

	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(which='major', length=8, labelsize=15, width=2)
	ax.tick_params(which='minor', length=4, width=2)
	plt.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False)

	ax.set_ylabel("Intensité relative (unité arbitraire)", fontsize="x-large")
	ax.set_xlabel(r"$\lambda$ (cm$^{-1}$)", fontsize="x-large")
	plt.legend()
	plt.tight_layout()
	#plt.show()
	plt.savefig("report/figures/echantillon1.png", bbox_inches="tight")

def echantillon2():
	e2 = pd.read_csv("data/2019-10-03/e2-100-6430-6465(bord)",
				names = ["lam", "cps"], sep="\t", skiprows=1)
	e2["lam"] = e2["lam"].apply(convert)
	mask = (e2.lam > 250)&(e2.lam < 550)
	plt.style.use('classic')
	fig, ax = plt.subplots()
	ax.plot(e2.lam[mask], e2.cps[mask], color="k", label="Données")
	ax.axvline(520, color="forestgreen", alpha=0.8, label="Si", linewidth=2, ls="--")
	ax.axvline(307, color="r", alpha=0.8, label="InP", linewidth=2, ls="--")
	ax.axvline(351, color="r", alpha=0.8, linewidth=2, ls="--")

	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(which='major', length=8, labelsize=15, width=2)
	ax.tick_params(which='minor', length=4, width=2)
	plt.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False)

	ax.set_ylabel("Intensité relative (unité arbitraire)", fontsize="x-large")
	ax.set_xlabel(r"$\lambda$ (cm$^{-1}$)", fontsize="x-large")
	plt.legend()
	plt.tight_layout()
	#plt.show()
	plt.savefig("report/figures/echantillon2(2).png", bbox_inches="tight")
	#axin = zoomed_inset_axes(ax, 0.9, loc='center right') # bbox_transform=ax.transAxes)  # zoom-factor: 2.5, location: upper-left
	#axin.plot(e2.lam[mask2], e2.cps[mask2])
	#mark_inset(ax, axin, loc1=2, loc2=4, fc="none", ec="0.5")
	#zoomed_inset_axes(ax, 10, bbox_to_anchor=(.8, 0, 1, 1), loc='center right',  bbox_transform=ax.transAxes)  # zoom-factor: 2.5, location: upper-left
	#mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


def echantillon3():
	e3 = pd.read_csv("data/2019-10-02/e3-100-6400-6500",
					names = ["lam", "cps"], sep="\t", skiprows=1)
	e3["lam"] = e3["lam"].apply(convert)
	mask = (e3.lam > 200)&(e3.lam < 360)
	plt.style.use('classic')
	fig, ax = plt.subplots()
	ax.plot(e3.lam[mask], e3.cps[mask], color="k", label="Données")

	plt.axvline(310, color="forestgreen", label=r'$SnS_2$', alpha=0.8, linewidth=2, ls="--")
	plt.axvline(230, color="r", alpha=0.8, linewidth=2, ls="--")
	plt.axvline(350, color="r", label=r"$TiS_2$", alpha=0.8, linewidth=2, ls="--")

	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(which='major', length=8, labelsize=15, width=2)
	ax.tick_params(which='minor', length=4, width=2)
	plt.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False)

	ax.set_ylabel("Intensité relative (unité arbitraire)", fontsize="x-large")	
	ax.set_xlabel(r"$\lambda$ (cm$^{-1}$)", fontsize="x-large")
	plt.legend()
	plt.tight_layout()
	#plt.show()
	plt.savefig("report/figures/echantillon3.png", bbox_inches="tight")

def sulfure():
	SS =  pd.read_csv("data/2019-10-03/souffre-6410-6540",
					names = ["lam", "cps"], sep="\t", skiprows=1)
	SAS = pd.read_csv("data/2019-10-03/souffre-6150-6250",
					names = ["lam", "cps"], sep="\t", skiprows=1)
	SS["lam"] = SS["lam"].apply(convert)
	SAS["lam"] = SAS["lam"].apply(convert)

	plt.style.use('classic')
	f, (ax, ax2) = plt.subplots(1, 2, sharey=True)
	ax.plot(SS.lam, SS.cps, color="k")
	ax.plot(SAS.lam, SAS.cps, color="k")
	ax2.plot(SS.lam, SS.cps, color="k")
	ax2.plot(SAS.lam, SAS.cps, color="k")

	ax2.set_xlim(200, 500)
	ax.set_xlim(-500, -200)

	ax.annotate("Spectre Anti-Stokes", xy=(0.1, 0.5), xycoords="axes fraction", fontsize="x-large")
	ax2.annotate("Spectre Stokes", xy=(0.2, 0.5), xycoords="axes fraction", fontsize="x-large")

	ax.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)

	d = .025 # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax.transAxes, color='k', clip_on=False,  lw=2.5)
	ax.plot((1-d,1+d), (-d,+d), **kwargs)
	ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d,+d), (1-d,1+d), **kwargs)
	ax2.plot((-d,+d), (-d,+d), **kwargs)


	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(which='major', length=8, labelsize=15, width=2, rotation=45)
	ax.tick_params(which='minor', length=4, width=2)
	ax.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False)

	ax2.xaxis.set_minor_locator(AutoMinorLocator())
	ax2.tick_params(which='major', length=8, labelsize=15, width=2, rotation=45)
	ax2.tick_params(which='minor', length=4, width=2)
	ax2.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False)

	ax.set_ylabel("Intensité relative (unité arbitraire)", fontsize="x-large")	
	f.text(0.5, 0, r"$\lambda$ (cm$^{-1}$)", ha='center', fontsize="x-large")
	plt.tight_layout()
	#plt.show()
	plt.savefig("report/figures/sulfure.png", bbox_inches="tight")

def sulfure_temperature():
	global results
	SS =  pd.read_csv("data/2019-10-03/souffre-6410-6540",
					names = ["lam", "cps"], sep="\t", skiprows=1)
	SAS = pd.read_csv("data/2019-10-03/souffre-6150-6250",
					names = ["lam", "cps"], sep="\t", skiprows=1)
	SS["lam"] = SS["lam"].apply(convert)
	SAS["lam"] = SAS["lam"].apply(convert)

	mean_SS = SS.cps[(SS.lam > 250)&(SS.lam < 300)].mean()
	mean_SAS = SAS.cps[(SAS.lam < -250)&(SAS.lam > -300)].mean()

	Ias = SAS.cps[(SAS.lam < -200)&(SAS.lam > -250)].max()
	Is = SS.cps[(SS.lam > 200)&(SS.lam < 250)].max()
	lamSS = SS.lam[SS.cps[(SS.lam > 200)&(SS.lam < 250)].idxmax()]
	lamSAS = SAS.lam[SAS.cps[(SAS.lam < -200)&(SAS.lam > -250)].idxmax()]
	lam = (lamSS - lamSAS)/2 / u.cm # moyenne des deux pics pour identifié lam en cm^-1, plus près de 218.
	T1 = temperature(Ias, Is, lam).cgs
	results.update({"TpremierPic": f"{T1.cgs:.0f}"})
	Ias = SAS.cps[(SAS.lam < -450)&(SAS.lam > -500)].max()
	Is = SS.cps[(SS.lam > 450)&(SS.lam < 500)].max()
	lamSS = SS.lam[SS.cps[(SS.lam > 450)&(SS.lam < 500)].idxmax()]
	lamSAS = SAS.lam[SAS.cps[(SAS.lam < -450)&(SAS.lam > -500)].idxmax()]
	lam = (lamSS - lamSAS)/2 / u.cm # moyenne des deux pics pour identifié lam en cm^-1, plus près de 218.
	T2 = temperature(Ias, Is, lam).cgs
	print(T1)
	print(T2)
	#results.update({"TdeuxiemePic": f"{T2.cgs:.0f}"})
	print( f"{np.array([T1.value, T2.value]).mean():.0f}")
	print(f"{np.array([T1.value, T2.value]).std():.2f}")

if __name__ == "__main__":


	#plasma = pd.read_csv("data/raman/fente_full_100.txt",
					#names = ["lam", "cps"], sep="\t", skiprows=1)



	#fentes()
	#write_to_tex()
	#echantillon1()
	#echantillon2()
	#echantillon3()
	#sulfure()
	sulfure_temperature()
	#write_to_tex()