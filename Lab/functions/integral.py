import numpy as np
# =============================================================================
# 2. Calcul de l'intégrale numérique selon la méthode trapèze
# =============================================================================
def trapeze_integral(func, a, b, npt=1000):
    """
    Intègre numériquement l'intégrande donnée dans l'intervalle $x\in[a,b]$
    avec la méthode des trapèzes avec un ordre $O(h^2)$. Le pas est  définie
    selon npt qui peut prendre n'importe quelle valeure entière.
    Intégrande calcul les valeurs de la fonction selon la maille définit
    dans le programme.
    Le programme retourne la valeure de l'intégrale et le pas utilisé.
    """
    x, h = np.linspace(a, b, npt, retstep=True)
    f = func(x)
    integral = h / 2 * (f[0] + f[-1] + 2 * f[1:-1].sum())
    return integral, h

# =============================================================================
# 3. Calcul de l'intégrale numérique selon la méthode Simpson composée
# =============================================================================

def simpson_integral(func, a ,b, npt):
    """
    Intègre numériquement l'intégrande donnée dans l'intervalle $x\in[a,b]$
    avec la méthode de Simpson avec un ordre $O(h^4)$. Le pas (h) de la maille
    est définie selon npt qui peut prendre n'importe quelle valeure entière.
    La fonction ajuste automatiquement le nombre de point pour que la maille
    possède un nombre pair d'intervalle. L'ajustement se fait à l'entier
    supérieur.
    Intégrande calcul les valeurs de la fonction selon la maille définit
    dans le programme.
    Le programme retourne la valeure de l'intégrale et le pas utilisé.
    """
    #===============
    n = npt
    while N % 2 != 1:
        n += 1
    #===============
    x, h = np.linspace(a, b, n, retstep=True)
    f = func(x)
    integral = h / 3 * (f[0] + f[-1] + 2 * (f[2:-2:2].sum() + 2 * f[1:-1:2].sum()))
    return integral, h

# =============================================================================
# 4. Calcul de l'intégrale selon la méthode de Boole
# =============================================================================
def boole_intergral(func, a, b, npt):
    """
    Intègre numériquement l'intégrande donnée dans l'intervalle $x\in[a,b]$
    avec la méthode de Boole avec un ordre $O(h^6)$. Le pas (h) de la maille
    est définie selon npt qui peut prendre n'importe quelle valeure entière.
    La fonction ajuste automatiquement le nombre de point pour que la maille
    possède un nombre d'intervalle divisible par 4. L'ajustement se fait à
    l'entier valide supérieur.
    Intégrande calcul les valeurs de la fonction selon la maille définit
    dans le programme.
    Le programme retourne la valeure de l'intégrale et le pas utilisé.
    """
    #===============
    n = npt
    while N % 4 != 1:
        n += 1
    #===============
    x, h = np.linspace(a, b, N, retstep=True)
    f = func(x)
    integral = 2 * h / 45 * (7 * (f[0] + f[-1]) + 32 * f[1:-1:2].sum() + 12 * f[2:-2:4].sum() + 14 * f[4:-4:4].sum())
    return integral, h
