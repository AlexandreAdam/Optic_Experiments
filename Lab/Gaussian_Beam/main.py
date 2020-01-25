import numpy as np
import pandas as pd
import os
from scipy.constants import pi
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt
import astropy.units as u
import cmath
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

LAM = (6328 * u.AA).to(u.micron).value

class ABCD:
	def __init__(self):
		self._abcd = np.array([[1, 0], [0, 1]])

	def transport(self, d):
		t = np.array([[1, d], [0, 1]])
		self._abcd = np.matmul(self._abcd, t)
		return self

	def refraction(self, f):
		r = np.array([[1, 0], [-1/f, 1]])
		self._abcd = np.matmul(self._abcd, r)
		return self

	def A(self):
		return self._abcd[0,0]

	def B(self):
		return self._abcd[0, 1]

	def C(self):
		return self._abcd[1, 0]

	def D(self):
		return self._abcd[1, 1]


def I(x, w, x0, Imax):
	return Imax / 2 * (1 - erf(2**(1/2) * (x - x0) / w))

def beam_waist(z, w0, z0=0):
	zR = pi * w0**2 / LAM
	return w0 * np.sqrt(1 + ((z-z0) / zR)**2)

def curvature(z, w0, z0):
	zR = pi * w0**2 / LAM
	zp = z - z0
	return zp * (1 + (zR / zp)**2)

def qinverse(z, w0, z0):
	return np.vectorize(complex)(1/curvature(z, w0, z0), 
		-LAM / pi / beam_waist(z, w0, z0)**2)

def plot_fit(data, popt, pcov, file):
	fig, ax = plt.subplots()

	xt = np.linspace(data.x.min(), data.x.max(), 1000)
	i = I(xt, *popt)
	i90 = i.max() * 0.9
	i10 = i.max() * 0.1
	plt.plot(data.x.values, data.I.values * 1000, "ko")
	plt.plot(xt, i * 1000, "r-")
	plt.xlabel(r"x [$\mu m$]", fontsize=16)
	plt.ylabel("I(x) [mA]", fontsize=16)
	plt.title(f"$z$= {float(file[2:]) / 100} cm", fontsize=16)
	plt.axhline(i90 * 1000, color="k")
	plt.annotate(r"90%$I_{max}$", (6000, (i90 + 0.02 * i90)*1000), fontsize=16)
	plt.annotate(r"10%$I_{max}$", (5600, (i10 + 0.1 * i10)*1000), fontsize=16)
	plt.annotate(fr"$w(z) = {popt[0]:.0f} \pm {pcov[0,0]**(1/2):.0f}$ $\mu$m", xy=(0.55, 0.6), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$x_0 = {popt[1]:.0f} \pm {pcov[1,1]**(1/2):.0f}$ $\mu$m",  xy=(0.55, 0.5), xycoords="axes fraction", fontsize=16)
	plt.axhline(i10 * 1000, color="k")
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(which='minor', axis='both', length=4)
	ax.tick_params(which='major', length=8)
	plt.savefig(f"report/figures/{file}_fit.png", bbox_inches="tight")

def plot_w(w, dw, z, popt, pcov, zmin=-20000):
	w0 = popt[0]
	dw0 = pcov[0, 0]**(1/2)
	z0 = popt[1]
	dz0 = pcov[1, 1]**(1/2)
	plt.figure()
	plt.style.use("classic")
	plt.errorbar(z / 1e4, w, yerr=dw, fmt="ko")
	zt = np.linspace(zmin, z.max(), 1000)
	plt.plot(zt / 1e4, beam_waist(zt, *popt), 'k-', lw=3, label=r"Ajustement sur $w(z)$")
	plt.xlabel("z [cm]", fontsize=16)
	plt.ylabel(r"w [$\mu m$]", fontsize=16)

def sans_lentille(files1, data_path_1,  position_du_laser, face_du_detecteur):
	zs = []
	ws = []
	dws = []
	for i, file in enumerate(files1):
		z = float(file[2:]) / 100
		#print(z)
		zs.append(z)
		data = pd.read_csv(data_path_1 + file, 
			names=["x", "I"],  #["mum", 'A']
			sep=r"\s+")
		popt, pcov = curve_fit(I, data.x.values, data.I.values, 
			p0=[200, 5500, 0.01])
		w = popt[0]
		dw = pcov[0, 0]**(1/2)
		#print(f"{w} +/- {dw}")
		ws.append(w)
		dws.append(dw)
		plot_fit(data, popt, pcov, file)

	z = (np.array(zs) * u.cm - position_du_laser - face_du_detecteur ).to(u.micron).value 
	w = np.array(ws) #deja en micron
	dw = np.array(dws)
	popt, pcov = curve_fit(beam_waist, z, w, p0=[300, 0])
	#print(f"w_0 = {popt[0]} +/- {pcov[0, 0]**(1/2)}")
	#print(f"z_0 = {popt[1]} +/- {pcov[1, 1]**(1/2)}")
	plot_w(w, dw, z, popt, pcov)
	zt = np.linspace(-20000, z.max(), 1000)
	w0 = popt[0]
	dw0 = pcov[0, 0]**(1/2)
	z0 = popt[1]
	dz0 = pcov[1, 1]**(1/2)
	plt.fill_between(zt / 1e4, 
		beam_waist(zt, w0 + dw0, z0 - dz0),
		beam_waist(zt, w0 - dw0, z0 + dz0),
		color="gray",
		alpha=0.6)
	plt.annotate(fr"$w_0 = {popt[0]:.0f} \pm {pcov[0, 0]**(1/2):.0f}$ $\mu$m",
				(0.1, 0.5), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$z_0 = {popt[1]/1e4:.0f} \pm {pcov[1, 1]**(1/2)/1e4:.0f}$ mm",
				(0.1, 0.35), xycoords="axes fraction", fontsize=16)
	plt.savefig("report/figures/w1.png")

	qz1 = -1j * (LAM/1e4) / pi / (w0/1e4)**2 #cm
	dqz1 = 1/qz1 * dw0 / w0
	return qz1, dqz1

def une_lentille(files2, data_path_2,  position_du_laser, face_du_detecteur, qz1):
	zs = []
	ws = []
	dws = []
	for file in files2:
		data = pd.read_csv(data_path_2 + file, 
			names=["x", "I"],  #["mum", 'A']
			sep=r"\s+")

		z = float(file[2:]) / 100
		#print(z)
		zs.append(z)
		popt, pcov = curve_fit(I, data.x.values, data.I.values, 
			p0=[200, 5500, 0.01])
		w = popt[0]
		dw = pcov[0, 0]**(1/2)
		#print(f"{w} +/- {dw}")
		ws.append(w)
		dws.append(dw)
		#plot_fit(data, popt, pcov, file)


	z = (np.array(zs) * u.cm - position_du_laser - face_du_detecteur).to(u.micron).value 
	w = np.array(ws) #deja en micron
	dw = np.array(dws)
	popt, pcov = curve_fit(beam_waist, z, w, p0=[300, 0])

	l1 = 43.86 * u.cm - position_du_laser - face_du_detecteur
	f1 = 17.5 * u.cm
	zt, h = np.linspace(z.min(), z.max(), 1000, retstep=True)
	qz2real = []
	qz2imag = []
	for z2 in zt:
		z2 /= 1e4 #cm
		abcd = ABCD()
		abcd = abcd.transport(z2 - l1.value).refraction(f1.value).transport(l1.value)
		qz2 = (abcd.C() +  abcd.D() * qz1)/(abcd.A() + abcd.B() * qz1)
		qz2real.append(qz2.real)
		qz2imag.append(qz2.imag)
	wth = np.sqrt(-(LAM/1e4) / np.array(qz2imag) / pi) * 1e4 #micron
	#print(f"w_0 = {popt[0]} +/- {pcov[0, 0]**(1/2)}")
	#print(f"z_0 = {popt[1]} +/- {pcov[1, 1]**(1/2)}")
	plot_w(w, dw, z, popt, pcov, z.min())
	plt.plot(zt/1e4, wth, 'k--', label="Théorique")
	w0 = popt[0]
	dw0 = pcov[0, 0]**(1/2)
	z0 = popt[1]
	dz0 = pcov[1, 1]**(1/2)
	plt.fill_between(zt / 1e4, 
		beam_waist(zt, w0 + dw0, z0 - dz0),
		beam_waist(zt, w0 - dw0, z0 + dz0),
		color="gray",
		alpha=0.6)
	plt.annotate(fr"$w_0 = {popt[0]:.0f} \pm {pcov[0, 0]**(1/2):.0f}$ $\mu$m",
				(0.4, 0.75), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$z_0 = {popt[1]/1e4:.1f} \pm {pcov[1, 1]**(1/2)/1e4:.1f}$ mm",
				(0.4, 0.65), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$f = {f1.to(u.mm).value}$ mm",
			(0.4, 0.55), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$z_0(th) = {zt[np.argmin(wth)]/1e4:.2f} \pm {h/1e4:.2f}$ cm",
			(zt[np.argmin(wth)]/1e4, wth.min() - 25), fontsize=16)
	plt.annotate(fr"$w_0(th) = {wth.min():.1f} \pm 0.1$ $\mu$m",
		(zt[np.argmin(wth)]/1e4, wth.min() - 45), fontsize=16)
	plt.xlim(z.min()/ 1e4 - 1, z.max() / 1e4 + 1)
	plt.legend(loc=9)
	plt.title(f"1 lentille à z = {l1.value:.1f} cm", fontsize=16)
	plt.savefig("report/figures/w2.png")
	qz1 = -1j * (LAM/1e4) / pi / (w0/1e4)**2 #cm
	dqz1 = 1/qz1 * dw0 / w0
	print(f"qz1lentille = {1/qz1} +/- {dqz1}")
	theta = (LAM/1e4) / pi / (w0/1e4)
	dtheta = theta * dw0 / w0
	print(f"theta = {theta* 180 / pi * 60} +/- {dtheta* 180 / pi * 60}")

	qz1 = -1j * (LAM/1e4) / pi / (wth.min()/1e4)**2 #cm
	dqz1 = 1/qz1 * 0.1 / w0
	print(f"qz1lentille(th) = {1/qz1} +/- {dqz1}")

	theta = (LAM/1e4) / pi / (wth.min()/1e4)
	dtheta = theta * 0.1 / w0
	print(f"theta(th) = {theta* 180 / pi * 60} +/- {dtheta* 180 / pi * 60} ")



def deux_lentilles(files3, data_path_3, position_du_laser, face_du_detecteur, qz1):

	zs = []
	ws = []
	dws = []
	for file in files3:
		data = pd.read_csv(data_path_3 + file, 
			names=["x", "I"],  #["mum", 'A']
			sep=r"\s+")

		z = float(file[2:]) / 100
		#print(z)
		zs.append(z)
		popt, pcov = curve_fit(I, data.x.values, data.I.values, 
			p0=[200, 5500, 0.01])
		w = popt[0]
		dw = pcov[0, 0]**(1/2)
		#print(f"{w} +/- {dw}")
		ws.append(w)
		dws.append(dw)
		#plot_fit(data, popt, pcov, file)

	z = (np.array(zs) * u.cm - position_du_laser - face_du_detecteur).to(u.micron).value 
	w = np.array(ws) #deja en micron
	dw = np.array(dws)
	popt, pcov = curve_fit(beam_waist, z, w, p0=[300, 0])

	l1 = 40 * u.cm - position_du_laser - face_du_detecteur
	f1 = 100* u.cm
	l2 = 47 * u.cm - position_du_laser - face_du_detecteur
	f2 = 50 * u.cm
	zt, h = np.linspace(z.min(), z.max(), 1000, retstep=True)
	qz2real = []
	qz2imag = []
	for z2 in zt:
		z2 /= 1e4 #cm
		abcd = ABCD()
		abcd = abcd.transport(z2 - l2.value).refraction(f2.value).transport(l2.value - l1.value).refraction(f1.value).transport(l2.value)
		qz2 = (abcd.C() +  abcd.D() * qz1)/(abcd.A() + abcd.B() * qz1)
		qz2real.append(qz2.real)
		qz2imag.append(qz2.imag)
	wth = np.sqrt(-(LAM/1e4) / np.array(qz2imag) / pi) * 1e4 #micron
	#print(f"w_0 = {popt[0]} +/- {pcov[0, 0]**(1/2)}")
	#print(f"z_0 = {popt[1]} +/- {pcov[1, 1]**(1/2)}")
	plot_w(w, dw, z, popt, pcov, z.min())
	plt.plot(zt/1e4, wth, 'k--', label="Théorique")
	w0 = popt[0]
	dw0 = pcov[0, 0]**(1/2)
	z0 = popt[1]
	dz0 = pcov[1, 1]**(1/2)
	plt.fill_between(zt / 1e4, 
		beam_waist(zt, w0 + dw0, z0 - dz0),
		beam_waist(zt, w0 - dw0, z0 + dz0),
		color="gray",
		alpha=0.6)
	plt.annotate(fr"$w_0 = {popt[0]:.0f} \pm {pcov[0, 0]**(1/2):.0f}$ $\mu$m",
				(0.4, 0.78), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$z_0 = {popt[1]/1e4:.1f} \pm {pcov[1, 1]**(1/2)/1e4:.1f}$ mm",
				(0.4, 0.69), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$f_1 = {f1.to(u.mm).value:.0f}$ mm",
			(0.4, 0.6), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$f_2 = {f2.to(u.mm).value:.0f}$ mm",
			(0.4, 0.51), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$z_0(th) = {zt[np.argmin(wth)]/1e4:.2f} \pm {h/1e4:.2f}$ cm",
			(zt[np.argmin(wth)]/1e4, wth.min() - 8), fontsize=16)
	plt.annotate(fr"$w_0(th) = {wth.min():.1f} \pm 0.1$ $\mu$m",
		(zt[np.argmin(wth)]/1e4, wth.min() - 15), fontsize=16)
	plt.xlim(z.min()/ 1e4 - 1, z.max() / 1e4 + 1)
	plt.legend(loc=9)
	plt.title(fr"2 lentille: $z_1$ = {l1.value:.1f} cm et $z_2$ = {l2.value:.1f} cm", fontsize=16)
	plt.savefig("report/figures/w3.png")
	qz1 = -1j * (LAM/1e4) / pi / (w0/1e4)**2 #cm
	dqz1 = 1/qz1 * dw0 / w0
	print(f"qz2lentille = {1/qz1} +/- {dqz1}")

	theta = (LAM/1e4) / pi / (w0/1e4)
	dtheta = theta * dw0 / w0
	print(f"theta = {theta * 180 / pi * 3600} +/- {dtheta * 180 / pi * 3600}")

	qz1 = -1j * (LAM/1e4) / pi / (wth.min()/1e4)**2 #cm
	dqz1 = 1/qz1 * 0.1 / w0
	print(f"qz2lentille(th) = {1/qz1} +/- {dqz1}")

	theta = (LAM/1e4) / pi / (wth.min()/1e4)
	dtheta = theta * 0.1 / w0
	print(f"theta(th) = {theta * 180 / pi * 3600} +/- {dtheta* 180 / pi * 60}")


def trois_lentilles(files4, data_path_4,  position_du_laser, face_du_detecteur, qz1):
	zs = []
	ws = []
	dws = []
	for file in files4:
		data = pd.read_csv(data_path_4 + file, 
			names=["x", "I"],  #["mum", 'A']
			sep=r"\s+")

		z = float(file[2:]) / 100
		#print(z)
		zs.append(z)
		popt, pcov = curve_fit(I, data.x.values, data.I.values, 
			p0=[200, 5500, 0.01])
		w = popt[0]
		dw = pcov[0, 0]**(1/2)
		#print(f"{w} +/- {dw}")
		ws.append(w)
		dws.append(dw)
		plot_fit(data, popt, pcov, file)

	z = (np.array(zs) * u.cm - position_du_laser - face_du_detecteur).to(u.micron).value 
	w = np.array(ws) #deja en micron
	dw = np.array(dws)
	popt, pcov = curve_fit(beam_waist, z, w, p0=[300, 0])

	l1 = 29.77 * u.cm - position_du_laser - face_du_detecteur
	f1 = 100 * u.cm
	l2 = 40.01 * u.cm - position_du_laser - face_du_detecteur
	f2 = -20 * u.cm
	l3 = 50 * u.cm - position_du_laser - face_du_detecteur
	f3 = 25 * u.cm
	zt, h = np.linspace(z.min(), z.max(), 1000, retstep=True)
	qz2real = []
	qz2imag = []
	for z2 in zt:
		z2 /= 1e4 #cm
		abcd = ABCD()
		abcd = abcd.transport(z2 - l3.value).refraction(f3.value).transport(l3.value - l2.value).refraction(f2.value).transport(l2.value - l1.value).refraction(f1.value).transport(l1.value)
		qz2 = (abcd.C() +  abcd.D() * qz1)/(abcd.A() + abcd.B() * qz1)
		qz2real.append(qz2.real)
		qz2imag.append(qz2.imag)
	wth = np.sqrt(-(LAM/1e4) / np.array(qz2imag) / pi) * 1e4 #micron
	#print(f"w_0 = {popt[0]} +/- {pcov[0, 0]**(1/2)}")
	#print(f"z_0 = {popt[1]} +/- {pcov[1, 1]**(1/2)}")
	plot_w(w, dw, z, popt, pcov, z.min())
	plt.plot(zt/1e4, wth, 'k--', label="Théorique")
	w0 = popt[0]
	dw0 = pcov[0, 0]**(1/2)
	z0 = popt[1]
	dz0 = pcov[1, 1]**(1/2)
	plt.fill_between(zt / 1e4, 
		beam_waist(zt, w0 + dw0, z0 - dz0),
		beam_waist(zt, w0 - dw0, z0 + dz0),
		color="gray",
		alpha=0.6)
	plt.annotate(fr"$w_0 = {popt[0]:.0f} \pm {pcov[0, 0]**(1/2):.0f}$ $\mu$m",
				(0.4, 0.78), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$z_0 = {popt[1]/1e4:.0f} \pm {pcov[1, 1]**(1/2)/1e4:.0f}$ mm",
				(0.4, 0.69), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$f_1 = {f1.to(u.mm).value:.0f}$ mm",
			(0.4, 0.6), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$f_2 = {f2.to(u.mm).value:.0f}$ mm",
			(0.4, 0.51), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$f_3 = {f3.to(u.mm).value:.0f}$ mm",
		(0.7, 0.51), xycoords="axes fraction", fontsize=16)
	plt.annotate(fr"$z_0(th) = {zt[np.argmin(wth)]/1e4:.2f} \pm {h/1e4:.2f}$ cm",
			(zt[np.argmin(wth)]/1e4 - 10, wth.min() + 20), fontsize=16)
	plt.annotate(fr"$w_0(th) = {wth.min():.1f} \pm 0.1$ $\mu$m",
		(zt[np.argmin(wth)]/1e4 - 10, wth.min() + 5), fontsize=16)
	plt.xlim(z.min()/ 1e4 - 1, z.max() / 1e4 + 1)
	plt.legend(loc=9)
	plt.title(fr"3 lentilles: $z_1$ = {l1.value:.1f} cm, $z_2$ = {l2.value:.1f} cm et $z_3$ = {l3.value:.1f}", fontsize=16)
	plt.savefig("report/figures/w4.png")
	qz1 = -1j * (LAM/1e4) / pi / (w0/1e4)**2 #cm
	dqz1 = 1/qz1 * dw0 / w0
	print(f"qz3lentille = {1/qz1} +/- {dqz1}")

	theta = (LAM/1e4) / pi / (w0/1e4)
	dtheta = theta * dw0 / w0
	print(f"theta = {theta * 180 / pi * 3600} +/- {dtheta * 180 / pi * 3600}")

	qz1 = -1j * (LAM/1e4) / pi / (wth.min()/1e4)**2 #cm
	dqz1 = 1/qz1 * 0.1 / w0
	print(f"qz3lentille(th) = {1/qz1} +/- {dqz1}")

	theta = (LAM/1e4) / pi / (wth.min()/1e4)
	dtheta = theta * 0.1 / w0
	print(f"theta(th) = {theta * 180 / pi * 3600} +/- {dtheta* 180 / pi * 60}")


def main():
	data_path_1 = "data/gaussian_beam/experience_sans_lentille/"
	data_path_2 = "data/gaussian_beam/lentille_f175mm/"
	data_path_3 =  "data/gaussian_beam/2lentilles_f1_1000_f2_500/"
	data_path_4 = "data/gaussian_beam/3lentilles3/"
	files1 = os.listdir(data_path_1)
	files2 = os.listdir(data_path_2)
	files3 = os.listdir(data_path_3)
	files4 = os.listdir(data_path_4)

	position_du_laser = 10.6 * u.cm #cm
	face_du_detecteur = 0.2  * u.cm#cm
	position_lentille1 = 43.86 * u.cm
	focal_length1 = 175 * u.mm
	position_lentille2 = 68.05 * u.cm
	focal_length2 = 250 * u.mm


	qz1, dqz1 = sans_lentille(files1, data_path_1,  position_du_laser, face_du_detecteur)
	print(f"qz0lentille = {1/qz1} +/- {dqz1}")

	#une_lentille(files2, data_path_2,  position_du_laser, face_du_detecteur, qz1)
	#deux_lentilles(files3, data_path_3,  position_du_laser, face_du_detecteur, qz1)
	#trois_lentilles(files4, data_path_4,  position_du_laser, face_du_detecteur, qz1)



def two_lens_focal_length(f1, f2, d):
	return f1 * f2 / (f1 + f2 - d)

def three_lens_focal_length(f1, f2, f3, d1, d2):
	efl12 = two_lens_focal_length(f1, f2, d1)
	return two_lens_focal_length(efl12, f3, d2 + d1)

if __name__ == '__main__':
	position_du_laser = 10.6 #* u.cm #cm
	face_du_detecteur = 0.2  #* u.cm#cm
	position_lentille1 = 43.86 #* u.cm
	focal_length1 = 17.5 #* u.mm
	position_lentille2 = 68.05 #* u.cm
	focal_length2 = 25.0 #* u.mm

	#position_lentille1 = 40 * u.cm
	#focal_length1 = 500 * u.mm
	#position_lentille2 = 47.5 * u.cm
	#focal_length2 = 1000* u.mm

	#print(f"f = {two_lens_focal_length(50, 100, 7)}")
	#print(f"f = {three_lens_focal_length(25 * u.cm, -20 * u.cm ,  100 * u.cm, 10 * u.cm, 10 * u.cm)}")
	main()
	#abcd = ABCD()
	# Note, abcd doit multiplier du dernier au premier (output à input)
		
