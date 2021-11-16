import networkx as nx
import time
from deffuant_polar import DeffuantModelPolar
from distribution_tools import normal_opinion
from distribution_tools import uniform_opinion
from distribution_tools import inverse_transform_sampling
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


# Initiating a opinions
N_nodes: int = 1000

# def gen_pdf(n_peaks, epsilon):
#     def pdf(x):
#         f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks-1)*np.pi*x)
#         f / np.trapz(f, x)
#         return f
#     return pdf


t0 = time.perf_counter()
# pdf = gen_pdf(3, 0.1)

initial_opinion = uniform_opinion(N_nodes)

# Initiate the model
model = DeffuantModelPolar(N_nodes, 0.5, 0.5)

# Set initial conditions in circled space
model.set_circled_opinion(initial_opinion)

model.show_opinion_distribution(model.get_unconverged_opinion())

# # Run one step
opinions = []
opinions.append(list(model.get_unconverged_opinion()))

for i in range(500000):
    new_opinion = model.one_step()
    opinions.append(list(new_opinion))

print('done')
model.show_opinion_distribution(opinions[-1])

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
circular_hist(ax[0], np.array(opinions[-1]), bins = 100)
plt.show()



