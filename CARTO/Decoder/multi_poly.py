from typing import List
import numpy as np
import matplotlib.pyplot as plt


class MultiPoly:
    def __init__(self, x: np.ndarray, Y: np.ndarray, poly_dim: int = 1):
        assert x.ndim == 1
        assert Y.ndim == 2
        assert x.shape[0] == Y.shape[0]
        self.poly_fits: List[np.polynomial.Polynomial] = []
        dim_amount = Y.shape[1]

        for lat_dim in range(dim_amount):
            poly = np.polynomial.Polynomial.fit(x, Y[:, lat_dim], poly_dim)
            self.poly_fits.append(poly)

        self.domain = np.array([np.min(x), np.max(x)])

    def get_vals(self, X: np.ndarray):
        return self.__call__(X)

    def get_domain_mean(self):
        x = np.mean(self.domain)
        return self.get_vals(x)

    def linspace(self, n: int = 50, domain=None):
        X = np.linspace(*(domain if domain else self.domain), num=n)
        return self(X)

    def __call__(self, X: np.ndarray):
        Ys = []
        for poly in self.poly_fits:
            Ys.append(poly(X))
        return np.stack(Ys, axis=0).T

    def get_plot(
        self,
        x: np.ndarray,
        Y: np.ndarray,
        domain=None,
        n_samples: int = 50,
        types: List[str] = [],
        markers=["v", "P", "d"],
    ):
        plt_dim = int(np.ceil(np.sqrt(len(self.poly_fits))))
        fig, axes = plt.subplots(
            plt_dim, plt_dim, figsize=(7, 7), sharex=True, sharey=True
        )
        for i, poly in enumerate(self.poly_fits):
            xx, yy = poly.linspace(n_samples, domain=domain if domain else self.domain)
            ax = axes[i // plt_dim][i % plt_dim]
            for type, marker in zip(set(types), markers):
                mask = np.array(types) == type
                ax.scatter(
                    x[mask],
                    Y[mask, i],
                    label=type,
                    marker=marker,
                    c=x[mask],
                    cmap="jet",
                )
            ax.plot(xx, yy, color="orange")
        return fig, axes
