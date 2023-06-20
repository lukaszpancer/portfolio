import numpy as np
from typing import Optional, Union, List


def gen_circles(
    n: int = 1000,
    no_circles: int = 4,
    radii: Union[List[float], float] = 5,
    noise: float = 0.2,
):
    """Generates `n` circles, all centered around (0, 0)

    Args:
        n (int, optional): Approximate number of total data points. Defaults to 1000.
        no_circles (int, optional): number of circles. Defaults to 2.
        radii (Union[List[float], float], optional): list of radii of the circles.
            When supplied with float the circles are 1 `radii` apart. Defaults to 1.
        noise (float, optional): Noise of the data points (Deviation from a perfect circle). Defaults to 0.2.
    """
    if type(radii) != list:
        radii = np.array([radii] * no_circles)
    radii = np.cumsum(radii)
    latent_space = np.random.random(n)
    radii = np.random.choice(radii, n)
    return np.array(
        [
            np.cos(2 * np.pi * latent_space) * radii + np.random.normal(0, noise, n),
            np.sin(2 * np.pi * latent_space) * radii + np.random.normal(0, noise, n),
        ]
    ).T


def gen_normal(
    n: int = 1000,
    weights: Optional[list[float]] = None,
    std: Union[float, list[float]] = 0.1,
    n_clust: int = 5,
):
    """Generates data from multivatiate (2d) normal distribution

    Args:
        n (int, optional): Approximate number of total data points. Defaults to 1000.
        std (float, optional): Standard deviation of clusters.
            When supplied with float, the value is set for all clusters. Defaults to 1.
        weights (list(float), optional): Defines how likely a cluster is to be assigned with a data point.
            Defaults to list of ones
        n_clust (int, optional): Number of generated clusters. Defaults to 3.
    """
    if hasattr(std, "__len__") and len(std) != n_clust:
        raise (
            ValueError(
                f"Got {std=}, but when supplied with arraylike its length must equal {n_clust=}"
            )
        )
    if type(std) == float:
        std = np.array(
            [
                std,
            ]
            * n_clust
        )
    if weights is None:
        weights = np.ones(shape=n_clust)
    # calculate n_clust means around a unit circle
    means = np.array(
        [
            (np.cos(2 * np.pi * x / n_clust), np.sin(2 * np.pi * x / n_clust))
            for x in range(n_clust)
        ]
    )
    print(means)
    cum_weights = np.cumsum(weights)
    sum_of_weights = np.sum(cum_weights)
    ret_data = []
    for mean, weight, s in zip(means, weights, std):
        cov = np.array([[s, 0], [0, s]])
        size = round(n * weight / sum_of_weights)
        ret_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=(size,)))
    return np.array(ret_data).reshape((-1, 2))
