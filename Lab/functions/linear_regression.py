from numpy import sqrt
import numpy as np


class Regression_lineaire:
    def __init__(self, x:np.array, y:np.array, dy=1):
        assert x.shape == y.shape
        self.x = x
        self.y = y
        self.dy = dy if isinstance(dy, np.ndarray) else np.ones(x.shape[0]) * dy
        self._delta = (1 / dy**2).sum() * (x**2 / dy**2).sum() - ((x / dy**2).sum())**2

    def pente(self):
        return \
            1 / self._delta \
                * ( \
                    (1 / self.dy**2).sum() \
                        * (self.x * self.y / self.dy**2).sum() \
                            - (self.x / self.dy**2).sum() \
                                 * (self.y / self.dy**2).sum() \
                    )

    def sigma_pente(self):
        return  sqrt(1 / self._delta * (1 / self.dy**2).sum())

    def intercept(self):
        return \
            1 / self._delta \
                * ( \
                    (self.x**2 / self.dy**2).sum() \
                        *(self.y / self.dy**2).sum() \
                             - (self.x / self.dy**2).sum() \
                                *(self.x * self.y / self.dy**2).sum() \
                    )

    def sigma_intercept(self):
        return  sqrt(1 / self._delta * (self.x**2 / self.dy**2).sum())

    def y_hat(self, x):
        return self.pente() * x + self.intercept()

    def sigma_y_hat(self, x):
        return sqrt(self.sigma_intercept()**2 + (self.sigma_pente() * x)**2 \
                    + 2 * self.covariance_pente_intercept() * self.sigma_intercept() * (self.sigma_pente() * x))

    def R_squared(self):
        """
        This is the R^2 test, which measures how much of the variance in y is explained by the model f.
        It runs from 1 to -1, both being good while 0 is very bad
        """
        return 1 - ((self.y - self.y_hat(self.x))**2).sum() / ((self.y - self.y.mean())**2).sum()

    def pearson_r(self):
        """
        This is a standard correlation test beteween x and y. A value of 1 or -1 implies that a linear model describes perfectly the data, 
        while a value of 0 implies there is no correlation between x and y
        """
        return ((self.x - self.x.mean()) * (self.y - self.error_weighted_average(self.y, self.dy))).sum() / self.x.std() / self.y.std()

    def chi_squared_reduced(self):
        return (((y - a - b * x) / dy)**2).sum() / (x.size - 2)

    def covariance_pente_intercept(self):    
        return - 1 / ((1 / self.dy)**2).sum() * self.x.mean() / ((self.x**2).mean() - (self.x).mean()**2)

    @staticmethod
    def error_weighted_average(x, sigma_x):
        return (x / sigma_x**2).sum() / (1 / sigma_x**2).sum()


if __name__ == "__main__":
    # Small tests to make sure this class works well
    x = np.linspace(10, 50000, 200)
    dy = np.random.normal(0, 1.5, 200)
    y = np.linspace(10, 50000, 200) + dy
    rl = Regression_lineaire(x, y, dy)

    print("Pente: %.1f +/- %.1e" % (rl.pente(), rl.sigma_pente()))
    print("Intercept %.1f +/- %.1e" % (rl.intercept(), rl.sigma_intercept()))
    assert rl.pente() >= 0
    assert abs(rl.pente() - 1) <= 0.1
    assert abs(rl.intercept()) <= 0.5, rl.intercept()
    assert rl.sigma_y_hat(5) <= 0.1, rl.sigma_y_hat(5) 
    assert abs(rl.y_hat(5) - 5) <= 0.1
    assert rl.pearson_r() >= 0.9
    assert rl.R_squared() >= 0.9
