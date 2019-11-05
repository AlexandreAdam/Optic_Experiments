import numpy as np
import astropy.units as u
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv # Bessel function
from scipy.optimize import curve_fit
import pandas as pd

bessel_zeros = np.array([[2.4048, 3.8317, 5.1356],[5.5201, 7.0156, 8.4172],[8.6537, 10.1735, 11.6198]])
def line(x, m, b):
	return m * x + b

def gaussian(x, A, mu, sigma):
	return A * np.exp(-(x - mu)**2 / sigma**2)

def number_of_modes(d, lam, numerical_aperture):
	return 0.5 * (pi * d / lam * numerical_aperture)**2

def critical_angle(n_1, n_2):
	return np.arcsin(n_2 / n_1) * 180 / pi

def numerical_aperture(n_1, n_2):
	return np.sqrt(n_1**2 - n_2**2)

def skip_distance(n_1, d, theta, n_0=1):
	# Typically, d around 50 microns
	return d * np.sqrt((n_1 / n_0 / np.sin(theta))**2 - 1)

def v_number(lam, a, numerical_aperture):
	return 2 * pi / lam * a * numerical_aperture

def s(l, m):
	return np.sqrt(bessel_zeros[m -1,l -1]**2 - l**2 - 1)

def s_zero():
	return np.sqrt(bessel_zeros[0, 1]**2 - 1)

def u_plus(l, m, v):
	u_c = bessel_zeros[m-1, l-1]
	#print(f"{l}{m}: {s(l, m)}")
	mask =  v > u_c
	ans = np.zeros(v.size) + v
	ans[mask] = u_c  * np.exp((np.arcsin(s(l, m) / u_c)  - np.arcsin(s(l, m) / v[mask]))/ s(l, m))
	return ans

def u_lzero(v):
	m=2
	u_c = bessel_zeros[0, 1]
	mask = v > u_c
	ans = np.zeros(v.size) + v
	ans[mask] = u_c  * np.exp((np.arcsin(s_zero() / u_c)  - np.arcsin(s_zero() / v[mask]))/ s_zero())
	return ans

def u_one(v):
	return v * (1 + 2**(1/2)) / (1 + (4 + v**4)**(1/4))

def b(l, m, v):
	if l == 0 and m==1:
		return 1 - u_one(v)**2 / v**2
	elif l == 0 and m == 2:
		return 1 - u_lzero(v)**2 / v**2
	else:
		return 1 - u_plus(l, m, v)**2 / v**2

def plot_b():
	plt.figure()

	v = np.linspace(0.1,10,10000)
	for l in range(0, 4):	
		for m in range(1, 3):
			if l == 0 and m==0:
				continue
			if l == 3 and m!=1:
				continue
			if l==2 and m==2:
				continue
			mask = b(l, m, v) > 0
			point_i = v[mask].size // 2
			bbox_props = dict(boxstyle="circle,pad=0.3", fc="white", ec="k", lw=2)
			plt.text(v[mask][point_i], b(l, m, v)[mask][point_i], f"{l}{m}", ha="center", va="center",
            				size=10, bbox=bbox_props)
			#plt.plot(v[mask][point_i], b(l, m, v)[mask][point_i], 'o',
        				#ms=circle_rad * 2, mec='b', mfc='none', mew=2)
			#plt.annotate(f"{l}{m}", xy=(v[mask][point_i], b(l, m, v)[mask][point_i]), size="large", textcoords='offset points')
			plt.plot(v[mask], b(l, m, v)[mask], "k-")
	plt.xticks([2.4048, 3.8317,  5.1356, 5.5201], ["2.4048", "3.8317",  "5.1356", "5.5201"], rotation=45)
	plt.ylabel(r"$b = 1 - \frac{u^2(v)}{v^2}$", fontsize="x-large")
	plt.xlabel("$v$", fontsize="x-large")
	plt.ylim(0, 1)
	plt.xlim(0, 7)
	plt.show()
	#plt.savefig("report/figures/theory.png", bbox_inches="tight")

def numerical_a_for_mon_B():
	print("MONOB")
	pixel_size = 5.75 * u.micron
	# MONO A
	#note: 0.75 means 0.75 * 1/10 pouces
	B1 = pd.read_csv("data/2019-10-23/monoB-rouge-1-LP01.txt", sep="\t", 
						skiprows=1, names=["pixel", "I"], header=None, index_col=False)
	B2 = pd.read_csv("data/2019-10-23/monoB-rouge-2-LP01.txt", sep="\t", 
						skiprows=1, names=["pixel", "I"], header=None, index_col=False)
	B15 = pd.read_csv("data/2019-10-23/monoB-rouge-1-50par1000-LP01.txt", sep="\t", 
						skiprows=1, names=["pixel", "I"], header=None, index_col=False)
	d = np.array([1, 2, 1.5]) * u.imperial.inch / 10
	d = d.to(u.mm)
	print(f"x = {d}")
	rs = []
	drs = []
	for data in [B1, B2, B15]:
		data.pixel *= 5.75
		popt, pcov = curve_fit(gaussian, xdata=data.pixel, ydata=data.I, 
									p0=[data.idxmax().I, data.idxmax().pixel, 10])
		r = popt[2] * (np.log(20))**(1/2) * u.micron
		dr = pcov[2, 2] * (np.log(20))**(1/2) * u.micron
		rs.append(r.to(u.mm).value)
		drs.append(dr.to(u.mm).value)
		print(f"r = {r.to(u.mm)} +/- {dr.to(u.mm)}")

	popt_x, pcov_x = curve_fit(line, xdata=d.to(u.mm).value, ydata=rs, p0=[0.2, 7 * 0.2], sigma=drs)
	print(f"N.A. = {popt_x[0]} +/- {pcov_x[0, 0]}")


def numerical_a_for_mon_A():
	print("MONOA")
	pixel_size = 5.75 * u.micron
	# MONO A
	#note: 0.75 means 0.75 * 1/10 pouces
	A075 = pd.read_csv("data/2019-10-15/monoA-LP01-rouge-0.75cm(2).txt", sep="\t", 
						skiprows=1, names=["pixel", "I"], header=None, index_col=False)
	A175 = pd.read_csv("data/2019-10-15/mono-LP01-rouge-1.75cm(2).txt", sep="\t", 
						skiprows=1, names=["pixel", "I"], header=None, index_col=False)
	A125 = pd.read_csv("data/2019-10-15/mono-LP01-rouge-1.25cm(2).txt", sep="\t", 
						skiprows=1, names=["pixel", "I"], header=None, index_col=False)
	d = np.array([0.75, 1.75, 1.25]) * u.imperial.inch / 10
	d = d.to(u.mm)
	print(f"x = {d}")
	rs = []
	drs = []
	for data in [A075, A175, A125]:
		data.pixel *= 5.75
		popt, pcov = curve_fit(gaussian, xdata=data.pixel, ydata=data.I, 
									p0=[data.idxmax().I, data.idxmax().pixel, 10])
		r = popt[2] * (np.log(3))**(1/2) * u.micron
		dr = pcov[2, 2] * (np.log(3))**(1/2) * u.micron
		rs.append(r.to(u.mm).value)
		drs.append(dr.to(u.mm).value)
		print(f"r = {r.to(u.mm)} +/- {dr.to(u.mm)}")


	popt_x, pcov_x = curve_fit(line, xdata=d.to(u.mm).value, ydata=rs, p0=[0.05, 7 * 0.2], sigma=drs)
	a = popt_x[1] / popt_x[0]
	print(np.array(rs) / (a * u.mm + d.to(u.mm)))
	print(f"N.A. = {popt_x[0]} +/- {pcov_x[0, 0]}")

def attenuation():
	L = 1.925 * u.m
	L2 = 68.6 * u.cm
	dL2 = 0.2 * u.cm
	dL = 0.4 * u.mm

	delta_L = L - L2
	p1_rouge = np.array([1.04, 1.05, 1.05, 1.05])
	p1_vert = np.array([0.38, 0.31, 0.42, 0.35, 0.33]) 
	p1_mauve = np.array([6.2, 6.1, ]) 

	p2_rouge = np.array([3.9, 4])
	p2_vert = np.array([0.49 + 0.54 + 0.53 + 0.49])
	p2_mauve = np.array([18.6, 18.5])


	kappa_rouge = 10  *np.log10(p2_rouge.mean() / p1_rouge.mean()) / delta_L.to(u.m)
	dkr = kappa_rouge * np.sqrt((dL + dL2)**2 / delta_L**2 + np.sqrt(p1_rouge.std()**2 / p1_rouge.mean()**2 + 0.05**2 / p2_rouge.mean()**2))

	kappa_vert = 10  *np.log10(p2_vert.mean() / p1_vert.mean()) / delta_L.to(u.m)
	dkv = kappa_vert * np.sqrt((dL + dL2)**2 / delta_L**2 + np.sqrt(p1_vert.std()**2 / p1_vert.mean()**2 + p2_vert.std()**2 / p2_vert.mean()**2))
	
	kappa_mauve = 10 * np.log10(p2_mauve.mean() / p1_mauve.mean()) / delta_L.to(u.m)
	dkm = kappa_mauve * np.sqrt((dL + dL2)**2 / delta_L**2 + np.sqrt(0.05**2 / p1_mauve.mean()**2 + 0.05**2 / p2_mauve.mean()**2))


	print(f"kappa rouge = {kappa_rouge} +/- {dkr}")
	print(f"kappa vert = {kappa_vert} +/- {dkv}")
	print(f"kappa kappa_mauve = {kappa_mauve} +/- {dkm}")

def main():
	lam_rouge = 632.8 * u.nm
	lam_bleue = 405 * u.nm
	lam_vert = 543.5 * u.nm
	lam_mauve = 445 * u.nm

	#mulitmode=============
	largeur_cercle = 6.7 
	distance = 14.4
	#=====================
	na = largeur_cercle / 2 / distance
	dna = na * np.sqrt(0.2**2 / largeur_cercle**2 + 0.1**2 / distance**2)
	print(f"numerical aperture {na} +/- {dna}")

	#monomode===============
	a_mono1 = 1.75
	a_mono2 = 2.2
	a_mono3 = 1.8

	rayons = np.array([1.75, 2.2, 1.8, 1.8]) * u.micron

	an_mono1 = 0.13
	an_mono2 = 0.13
	an_mono3 = 0.2
	print(2 * 52.5 * u.micron / lam_rouge * 0.22 + 1)
	print(2 * 1.75 * u.micron / lam_rouge * 0.22 + 1)
	print(2 * 2.2 * u.micron / lam_rouge * 0.22 + 1)
	print(2 * 1.8 * u.micron / lam_rouge * 0.22 + 1)

	numerical_apertures = np.array([an_mono1, an_mono2, an_mono3, 0.33])
	print(f"N.A = {str(numerical_apertures)}")
	print(f"core radius = {str(rayons)}")
	v_rouge = v_number(lam_rouge, rayons, numerical_apertures)
	print(f"{lam_rouge}: V numbers = {str(v_rouge.cgs)}")
	v_vert = v_number(lam_vert, rayons, numerical_apertures)
	print(f"{lam_vert}: V Numbers = {str(v_vert.cgs)}")
	v_bleue = v_number(lam_bleue, rayons, numerical_apertures)
	print(f"{lam_bleue}: V numbers = {str(v_bleue.cgs)}")

	v_mauve = v_number(lam_mauve, rayons, numerical_apertures)
	print(f"{lam_mauve}: V numbers = {str(v_mauve.cgs)}")

def extra():
	d = 8 * u.cm
	aprime = 5 * u.mm
	gamma = 40
	o = d / gamma
	
	print(f"a = {(aprime / gamma).to(u.micron)}")
	print(f"a' should be = {gamma * (105 *u.micron).to(u.mm)}")

if __name__ == "__main__":
	# glass/plastic: n_1 = 1.46 and n_2 = 1.40
	#print(critical_angle(1.46, 1.4))
	#print(numerical_aperture(1.46, 1.4))

	#numerical_a_for_mon_A()
	#numerical_a_for_mon_B()
	#main()
	attenuation()
	#plot_b()

