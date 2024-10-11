import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random
from matplotlib.collections import LineCollection
import warnings
from pandas.errors import SettingWithCopyWarning
import pandas as pd


FONT_SIZE = 12
PALETTE = sns.color_palette("viridis_r", as_cmap=True)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter('ignore', np.RankWarning)


def save_figure(name='figure'):
    plt.savefig(f'{name}.png', format="png", dpi=500)
    plt.show()


def new_plot(palette='pastel', style='white'):
    sns.set_style(style)
    sns.set_palette(palette)


def linear_regression(df, X, y):
    lr = LinearRegression().fit(df[X], df[y])
    return lr.coef_, lr.intercept_, lr.score(df[X], df[y])


def linear_regression_bic(df, X, y):
    lr = LinearRegression().fit(df[X], df[y])
    predictions = lr.predict(df[X])
    rss = mean_squared_error(df[y], predictions, squared=False) * len(y)
    n = len(df[y])
    k = df[X].shape[1] + 1  # Number of parameters (number of features + 1 for intercept)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


def log_linear_regression(df, X, y):
    lr = LinearRegression().fit(np.log(df[X]), np.log(df[y]))
    return lr.coef_, lr.intercept_, lr.score(np.log(df[X]), np.log(df[y]))


def labels(x='Served Population (inhabitants)', y='Sewer Length (km)'):
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)


def plot_best_fit(dataframe, x_indicator, y_indicator, color, label='', width=0.7,
                  alpha=0.4, ls='-', ax=None, log=True):
    if log:
        coefficients = exponential_best_fit(np.array(dataframe[x_indicator]), np.array(dataframe[y_indicator]))
        x_axis = np.array(dataframe[x_indicator])
        y_axis = np.array((x_axis ** coefficients[0]) * np.exp(coefficients[1]))
    else:
        coefficients = best_fit(np.array(dataframe[x_indicator]), np.array(dataframe[y_indicator]))
        x_axis = np.array(dataframe[x_indicator])
        y_axis = np.array((x_axis * coefficients[0]) + coefficients[1])
    if ax is None:
        plt.plot(x_axis, y_axis, color=color, alpha=alpha, lw=width, label=label, ls=ls)
    else:
        ax.plot(x_axis, y_axis, color=color, alpha=alpha, lw=width, label=label, ls=ls)


def best_fit(x, y):
    coefficients = np.polyfit(x, y, 1)
    return coefficients


def exponential_best_fit(x, y):
    coefficients = np.polyfit(np.log(x), np.log(y), 1)
    return coefficients


def exponential_regression_coefficients(df, X, y):
    lr = LinearRegression().fit(np.log(df[X]), np.log(df[y]))
    return lr.coef_, lr.intercept_


def add_color_bar(ax, title='k for the city'):
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
    sm.set_array([])
    color_bar = inset_axes(ax,
                           width="3%",
                           height="30%",
                           loc='upper left',
                           bbox_to_anchor=(0.855, -0.54, 1.0, 0.85),
                           bbox_transform=ax.transAxes)
    try:
        ax.get_legend().remove()
    except:
        pass
    cb = ax.figure.colorbar(sm, cax=color_bar, ticks=[0.2, 0.4, 0.6, 0.8])
    cb.outline.set_visible(False)
    cb.ax.set_title(title, fontsize=11)


def longitudinal_track_colormap(df, x_axis='Total Population', y_axis='Sewer Length', filter_fn=lambda x, y: False,
                                cutoff=0, percentage=1):
    fig, ax = plt.subplots()
    # Create dictionaries
    x_var, y_var, k = dict(), dict(), dict()
    for index, row in df.iterrows():
        try:
            x_var[row["City Code"]].append(row[x_axis])
            y_var[row["City Code"]].append(row[y_axis])
            k[row["City Code"]].append(row["Percent Served"])
        except:
            x_var[row["City Code"]] = [row[x_axis]]
            y_var[row["City Code"]] = [row[y_axis]]
            k[row["City Code"]] = [row["Percent Served"]]

    # Filtering for bad data and plotting
    for key in x_var:
        if 0 in x_var[key] or 0 in y_var[key]:
            pass
        elif 'nan' in x_var[key] or 'nan' in y_var[key]:
            pass
        elif filter_fn(x_var[key], y_var[key]) is True:
            pass
        elif len(x_var[key]) < cutoff:
            pass
        elif random.random() < 1 - percentage:
            pass
        else:
            x, y, attended = np.array(x_var[key]), np.array(y_var[key]), np.array(k[key])
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, 1)
            lc = LineCollection(segments, cmap='viridis_r', norm=norm)
            lc.set_array(attended)
            lc.set_linewidth(1.2)
            ax.add_collection(lc)
    return ax


def get_longitudinal_coefficient(df, x_axis='Total Population', y_axis='Sewer Length', cutoff=8):
    # Create dictionaries
    x_var, y_var, x_var_, y_var_ = {}, {}, {}, {}
    for index, row in df.iterrows():
        if row["City Code"] in x_var:
            x_var[row["City Code"]].append(row[x_axis])
            y_var[row["City Code"]].append(row[y_axis])
        else:
            x_var[row["City Code"]] = [row[x_axis]]
            y_var[row["City Code"]] = [row[y_axis]]

    # Filtering for bad data
    for key in x_var:
        if 0 in x_var[key] or 0 in y_var[key]:
            pass
        elif 'nan' in x_var[key] or 'nan' in y_var[key]:
            pass
        else:
            x_var_[key] = x_var[key]
            y_var_[key] = y_var[key]

    coefficients = {'City Code': [], 'coeff': [], 'A_i': []}

    # Finding coeff
    for key in x_var_:
        if len(x_var_[key]) >= cutoff:
            fit = np.polyfit(np.log(x_var_[key]), np.log(y_var_[key]), 1)
            coefficients['City Code'].append(key)
            coefficients['coeff'].append(fit[0])
            coefficients['A_i'].append(fit[1])

    return coefficients


def add_flare_color_bar(ax, title='Input year'):
    norm = plt.Normalize(2005, 2020)
    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    sm.set_array([])
    color_bar = inset_axes(ax,
                           width="3%",
                           height="30%",
                           loc='upper left',
                           bbox_to_anchor=(0.855, -0.54, 1.0, 0.85),
                           bbox_transform=ax.transAxes)
    ax.get_legend().remove()
    cb = ax.figure.colorbar(sm, cax=color_bar, ticks=[2007, 2012, 2017])
    cb.outline.set_visible(False)
    cb.ax.set_title(title, fontsize=11)


def load_datasets():
    snis_sewer = pd.read_csv('../data/clean/snis_sewer.csv')
    snis_water = pd.read_csv('../data/clean/snis_water.csv')
    ibnet_sewer = pd.read_csv('../data/clean/ibnet_sewer.csv')
    ibnet_water = pd.read_csv('../data/clean/ibnet_water.csv')
    return snis_sewer, snis_water, ibnet_sewer, ibnet_water


def longitudinal_scaling(df, scaling_type, last_year, save=False, length_variable='Sewer Length', path=''):
    random.seed(0)
    df = df.sort_values(by=['Year'])
    df['Length'] = df[length_variable]
    new_plot()
    inset_color = (0.128729, 0.563265, 0.551229)
    inset_alpha = 0.1

    if scaling_type == "total":
        ax = longitudinal_track_colormap(df, cutoff=10, percentage=0.25)
        n_coefficients = exponential_regression_coefficients(df[df['Year'] == last_year],
                                                             ['Total Population'], ['Length'])[0]
        plot_best_fit(df[df['Year'] == last_year], 'Total Population', 'Length', 'k', width=1.4, alpha=1)
        ax.set_title('$Y = A\,N^{α}$', y=1.0, pad=-14, fontsize=12)
        plt.text(10 ** 6.60, 10 ** 3.20, f'$β = {n_coefficients[0][0]:.2f}$', fontsize=11)

        # Plot adjustments
        ax.set_xlim(0.4 * 10 ** 2, 0.43 * 10 ** 8)
        ax.set_ylim(0.4 * 10 ** 0, 1.2 * 10 ** 5)
        sns.despine()
        plt.xscale('log')
        plt.yscale('log')
        labels('Total Population (inhabitants)', f'{length_variable} (km)')
        add_color_bar(ax)

        # get longitudinal coefficients and their mean
        alphas = get_longitudinal_coefficient(df, cutoff=10)
        alphas = pd.DataFrame.from_dict(alphas)
        mean_alpha = alphas.describe().loc['mean', 'coeff']

        # Inset plot
        n_coefficients = exponential_regression_coefficients(df[df['Year'] == last_year], ['Total Population'],
                                                             ['Length'])[0]
        ax2 = ax.inset_axes([0.12, 0.73, 0.27, 0.25])
        sns.kdeplot(x='coeff', data=alphas, color=inset_color, alpha=inset_alpha, fill=True, ax=ax2)
        ax2.set_ylim(0, 0.6)
        ax2.set_xlim(-10, 20)
        ax2.set_xlabel('$α$ for the city')
        ax2.axvline(x=n_coefficients[0][0], color='k', ls='--', lw=1, alpha=0.75)
        ax2.text(2.4, 0.25, f'{n_coefficients[0][0]:.2f}')
        ax2.text(6.0, 0.07, '$\overline{α}$' + f' = {mean_alpha:.2f}')

        if save:
            save_figure(path + 'l-population_model')
        else:
            plt.show()

        return alphas

    elif scaling_type == "served":
        ax = longitudinal_track_colormap(df, 'Served Population', cutoff=10, percentage=0.25)
        m_coefficients = exponential_regression_coefficients(df[df['Year'] == last_year],
                                                             ['Served Population'], ['Length'])[0]

        plot_best_fit(df[df['Year'] == last_year], 'Served Population', 'Length', 'k', width=1.4, alpha=1)
        ax.set_title('$Y = A\,M^{α}$', y=1.0, pad=-14, fontsize=12)
        plt.text(10 ** 6.60, 10 ** 3.20, f'$β = {m_coefficients[0][0]:.2f}$', fontsize=11)

        # Plot adjustments
        ax.set_xlim(0.4 * 10 ** 2, 0.43 * 10 ** 8)
        ax.set_ylim(0.4 * 10 ** 0, 1.2 * 10 ** 5)
        sns.despine()
        plt.xscale('log')
        plt.yscale('log')
        labels('Served Population (inhabitants)', f'{length_variable} (km)')
        add_color_bar(ax)

        # get longitudinal coefficients and their mean
        alphas2 = get_longitudinal_coefficient(df, 'Served Population', cutoff=10)
        alphas2 = pd.DataFrame.from_dict(alphas2)
        mean_alpha2 = alphas2.describe().loc['mean', 'coeff']

        # Inset plot
        ax2 = ax.inset_axes([0.12, 0.73, 0.27, 0.25])
        sns.kdeplot(x='coeff', data=alphas2, color=inset_color, alpha=inset_alpha, fill=True, ax=ax2)
        ax2.set_ylim(0, 0.6)
        ax2.set_xlim(-10, 20)
        ax2.set_xlabel('$α$ for the city')
        ax2.axvline(x=m_coefficients[0][0], color='k', ls='--', lw=1, alpha=0.75)
        ax2.text(2.7, 0.25, f'{m_coefficients[0][0]:.2f}')
        ax2.text(5.0, 0.07, '$\overline{α}$' + f' = {mean_alpha2:.2f}')
        if save:
            save_figure(path + 'l-development_model')
        else:
            plt.show()

        return alphas2

    else:
        print("Invalid scaling type")
        return None


def snis_cross_sectional_scaling(df, scaling_type, save=False, length_variable='Sewer Length', variant=''):
    new_plot()
    ax = None

    df['Length'] = df[length_variable]

    if scaling_type == "total":
        coefficients, intercept, score = log_linear_regression(df, ['Total Population'], ['Length'])
        ax = sns.scatterplot(y='Length', x='Total Population', data=df.sample(frac=1, random_state=0),
                             hue='Percent Served', alpha=0.5, palette=PALETTE)
        # ax.set_title('Traditional: $Y = Y_0\,N^β$', y=1.0, pad=-14, fontsize=FONT_SIZE)
        labels('Total Population (inhabitants)', f'{length_variable} (km)')
        plt.text(10 ** 6.67, 10 ** 2.65, f'$β = {coefficients[0][0]:.2f}$\n$R^2 = {score:.2f}$', fontsize=11)
        plot_best_fit(df, 'Total Population', 'Length', 'k', width=1.4, alpha=1)

    elif scaling_type == "served":
        coefficients, intercept, score = log_linear_regression(df, ['Served Population'], ['Length'])
        ax = sns.scatterplot(y='Length', x='Served Population', data=df.sample(frac=1, random_state=0),
                             hue='Percent Served', alpha=0.5, palette=PALETTE)
        # ax.set_title('Generalized: $Y = Y_0\,M^{β}$', y=1.0, pad=-14, fontsize=FONT_SIZE)
        labels('Served Population (inhabitants)', f'{length_variable} (km)')
        plt.text(10 ** 6.60, 10 ** 2.95, f'$β = {coefficients[0][0]:.2f}$\n$R^2 = {score:.2f}$', fontsize=11)
        plot_best_fit(df, 'Served Population', 'Length', 'k', width=1.4, alpha=1)

    else:
        print("Invalid scaling type")
        return 0

    ax.set(xscale='log', yscale='log')
    plt.xlim(0.4 * 10 ** 2, 0.43 * 10 ** 8)
    plt.ylim(0.4 * 10 ** 0, 1.2 * 10 ** 5)
    sns.despine()
    add_color_bar(ax)
    if save:
        save_figure(f'length_vs_{scaling_type}_with_k{variant}', '2020')
    else:
        plt.show()

    return coefficients[0][0]


def ibnet_cross_sectional_scaling(df, scaling_type, save=False, length_variable='Sewer Length', variant=''):
    new_plot()
    ax = None

    df['Length'] = df[length_variable]

    if scaling_type == "total":
        coefficients, intercept, score = log_linear_regression(df, ['Total Population'], ['Length'])
        ax = sns.scatterplot(y='Length', x='Total Population', data=df.sample(frac=1, random_state=0),
                             hue='Percent Served', alpha=0.5, palette=PALETTE)
        ax.set_title('Traditional: $Y = Y_0\,N^β$', y=1.0, pad=-14, fontsize=FONT_SIZE)
        labels('Total Population (inhabitants)', f'{length_variable} (km)')
        plt.text(10 ** 2, 10 ** 3.25, f'$β = {coefficients[0][0]:.2f}$\n$R^2 = {score:.2f}$', fontsize=11)
        plot_best_fit(df, 'Total Population', 'Length', 'k', width=1.4, alpha=1)

    elif scaling_type == "served":
        coefficients, intercept, score = log_linear_regression(df, ['Served Population'], ['Length'])
        ax = sns.scatterplot(y='Length', x='Served Population', data=df.sample(frac=1, random_state=0),
                             hue='Percent Served', alpha=0.5, palette=PALETTE)
        ax.set_title('Generalized: $Y = Y_0\,M^{β}$', y=1.0, pad=-14, fontsize=FONT_SIZE)
        labels('Served Population (inhabitants)', f'{length_variable} (km)')
        plt.text(10 ** 2, 10 ** 3.25, f'$β = {coefficients[0][0]:.2f}$\n$R^2 = {score:.2f}$', fontsize=11)
        plot_best_fit(df, 'Served Population', 'Length', 'k', width=1.4, alpha=1)

    else:
        print("Invalid scaling type")
        return 0

    ax.set(xscale='log', yscale='log')
    plt.xlim(0.4 * 10 ** 2, 0.43 * 10 ** 8)
    plt.ylim(0.4 * 10 ** 0, 1.2 * 10 ** 5.5)
    sns.despine()
    add_color_bar(ax)
    if save:
        save_figure(f'../ibnet_figures/{scaling_type}_with_k_{variant}', '2020')
    else:
        plt.show()

    return coefficients[0][0]