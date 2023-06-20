import plotly.express as px
from .datasets import gen_circles, gen_normal
from collections import namedtuple
from sklearn.datasets import make_moons, load_iris
import numpy as np
import pandas as pd

# list of some predefined dataset generators
dataset = {
    "circles": gen_circles,
    "normal": gen_normal,
    "moons": make_moons,
    "iris": load_iris,
}


def generate_clustering_plot(data: np.ndarray, labels: np.ndarray, title: str = ""):
    """Generates a plotly plot of the clustering

    Args:
        data (np.ndarray): Data containing the clusters of dimension (n, d), where n is the number of data points and d is the dimension of the data points (must be 2 or 3)
        labels (np.ndarray): Labels of the clusters, must be of length n
        title (str, optional): The title of the plot. Defaults to "".
    """
    # check if the data is 2d or 3d
    if data.shape[1] > 3:
        raise ValueError(
            f"Data must be 2d or 3d, but got {data.shape[1]} dimensions instead"
        )
    columns = [dim_name for dim_name, _ in zip(["x", "y", "z"], data[1])]
    columns_kwargs = {k: k for k in columns}

    df = pd.DataFrame(data, columns=columns)
    df["label"] = labels
    # create the plotly figure
    fig = px.scatter(
        df,
        **columns_kwargs,
        color="label",
        title=title,
        template="plotly_white",
        width=600,
        height=600,
    )
    return fig


def get_plot():
    return generate_clustering_plot(
        np.array([1, 2, 3]).reshape(3, 1), np.array([1, 2, 1])
    )
