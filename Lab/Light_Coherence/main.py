import astropy.units as u
import numpy as np
DELTAD = 1 * u.micron

def nSF6(lam):
	return 1 + (71517 + 5e-6 * u.nm**2 / lam**2) * 1e-8

def nH2O(lam):
	return 1.3199 + 6878 * u.nm**2 / lam**2 - 1.132e9 * u.nm**4 / lam**4 + 1.11e14*u.nm**6 / lam**6

def nHe(lam):
	return 1 + 0.01470091 / u.nm**2 / (423.98 /u.nm**2 - lam**(-2))

def nAir(lam):
	return 1 + 0.05792105 / u.nm**2 / (238.0185 / u.nm**2 - lam**(-2)) + \
			0.00167917 / u.nm**2 / (57.362/u.nm**2 - lam**(-2))

def calibration():
	lam  = 6328 * u.AA
	delta_m = 100
	delta_d_true = (lam * delta_m / 2).to(u.micron)
	print(f"{delta_d_true}")
	delta_d = 15.5 * u.micron
	facteur = (delta_d_true / 15.5 / u.micron + delta_d_true / 17 / u.micron)/2
	print(f"facteur calibration = {facteur:.2f}")
	return facteur

def mesure_lam_HG(facteur):
	delta_m = 110
	delta_d = 15.5 * u.micron * facteur # x2 est le facteur de calibrage
	lam = (2  * delta_d / delta_m).to(u.micron)
	error_lam = lam * DELTAD / delta_d
	print(f"deltad = {delta_d}")
	print(f"lam Hg= {lam:.3f} +/- {error_lam:.0e}")
	#print(f"delta_d th Hg= {(5461 * u.AA * delta_m / 2 /  facteur).to(u.micron):.0f}")

	delta_lam = 0.597 * u.nm
	print(f"lc Hg= {(lam**2 / delta_lam).to(u.mm)} +/- {(lam**2 / delta_lam).to(u.mm) * error_lam / lam}")
	print(f"")

def mesure_lam_NA(facteur):
	delta_m = 100
	lam_th1 = 5889.95 * u.AA
	lam_th2 = 5895.924 * u.AA
	lam_moy = lam_th1 + lam_th2
	lam_moy /= 2
	print(f"delta_d th NA = {(lam_moy * delta_m / 2 / facteur).to(u.micron):.0f}")
	delta_d = 15 * u.micron * facteur
	lam = (2  * delta_d / delta_m).to(u.micron)
	error_lam = lam * DELTAD / delta_d
	print(f"lam NA= {lam:.3f} +/- {error_lam}")
	print(f"lam_moy_th NA= {lam_moy:.0f}")

	lc_exp = 160 * u.micron * facteur
	dlc = DELTAD * 5
	print(f"lc NA exp = {lc_exp} +/- {dlc}")
	dlam = lam**2 / lc_exp
	dlam = dlam.to(u.nm)
	ddlam = dlam * (dlc**2 / lc_exp**2 + 4 * error_lam**2 / lam**2)**(1/2)
	print(f"delta_lam NA = {dlam} +/- {ddlam}")

	delta_lam = abs(lam_th1 - lam_th2)
	print(f"delta_lam NA th= {delta_lam}")
	lcth = (lam_moy**2 / delta_lam/ facteur).to(u.micron)
	print(f"lctheorique NA= {lcth:.2f}")


def mesure_n():
	d = 7.2 * u.cm
	delta_m = 1/3 * (8  + 9 + 8)
	ddm = 1
	lam = 6328 * u.AA
	OPD = lam * delta_m / 2 # optical path length
	dO = OPD * (ddm**2 / delta_m**2 + 0.1**2 * u.cm**2 / d**2)**(1/2)
	n = OPD / d + 1
	dn = (n - 1) * dO / OPD
	print(f"n(He) = {n:.7f} +/- {dn:.2e}")
	delta_m = 61
	ddm = 2
	OPD = lam * delta_m / 2 # optical path length
	n = OPD / d + 1
	dO = OPD * ddm / delta_m
	dn = (n - 1)* dO / OPD
	print(f"n(Air) = {n:.7f} +/- {dn:.2e}")
	delta_m = 195
	OPD = lam * delta_m / 2 # optical path length
	n = OPD / d + 1
	ddm = (delta_m)**(1/2)
	print(ddm)
	dO = OPD * ddm / delta_m
	dn = (n - 1) * dO / OPD
	print(f"n(SF6) = {n:.7f} +/- {dn:.2e}")

	delta_m = 15
	ddm = 1
	lam = 405 * u.nm
	OPD = lam * delta_m / 2 # optical path length
	n = OPD / d + 1
	dO = OPD * ddm / delta_m
	dn = (n - 1) * dO / OPD
	print(f"n(He)405 = {n:.7f} +/- {dn:.2e}")


def mesure_nlam():
	lam1 = 632.8 * u.nm
	lam2 = 515 * u.nm
	lam3 = 543.5 * u.nm
	print(nAir(lam1))
	print(nAir(lam2))
	print(nAir(lam3))

def lumiere_blanche(facteur):
	lam = np.array([450, 500, 550, 600, 650, 700]) * u.nm
	lc = np.array([9, 10, 14, 16, 17, 17]) * facteur * u.micron
	dlc = 5 * u.micron
	deltalam = (lam**2 / lc).to(u.nm)
	print(lc)
	print(f"dlc = {dlc}")
	print(deltalam)
	print(f"error dlam = {deltalam * dlc / lc}")


if __name__ == '__main__':
	facteur = calibration()
	lumiere_blanche(facteur)
	#mesure_lam_HG(facteur)
	#mesure_lam_NA(facteur)
	mesure_n()
	#mesure_nlam()