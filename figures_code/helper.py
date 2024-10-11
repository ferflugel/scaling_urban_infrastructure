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
from sklearn.metrics import r2_score
import pickle
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from tqdm.notebook import tqdm

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter('ignore', np.RankWarning)

FONT_SIZE = 12
PALETTE = sns.color_palette("viridis_r", as_cmap=True)
PURPLE = (0.69531250, 0.66796875, 0.82031250)
ORANGE = (0.98828125, 0.71875000, 0.38671875)


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


def length_predictions(df, last_year=2020, save=False, length_variable='Sewer Length', axes_limits=(150, 400),
                       name_suffix='', year_labels=((180, 205), (239, 272), (293, 319), (353, 377))):

    # Create list for error
    error, error_year = [], []
    predicted_growth = []
    for i in range(15):
        error.append([])
        error_year.append([])
        predicted_growth.append([])

    # Run simulations
    x, y, hue = [], [], []
    for ty in range(2005, last_year + 1):
        real_value = df[df['Year'] == ty][length_variable].sum()
        for iy in range(2005, ty):
            prediction = bootstrap_prediction(df, iy, ty, 0, 1, length_variable)
            x.append(prediction)
            y.append(real_value)
            hue.append(iy)
            error[ty - iy - 1].append(abs(prediction - real_value) / real_value)
            error_year[ty - iy - 1].append(iy)
            predicted_growth[ty - iy - 1].append((prediction - df[df['Year'] == iy][length_variable].sum()) /
                                                 df[df['Year'] == iy][length_variable].sum())

    # Plot the simulations
    new_plot()
    c = sns.color_palette("flare", as_cmap=True)
    ax = sns.scatterplot(y=np.array(x)/1000, x=np.array(y)/1000, hue=hue, palette=c, alpha=0.75, zorder=10)

    # Style and saving plot
    sns.despine()

    r2 = r2_score(np.array(y)/1000, np.array(x)/1000)

    # Plot x = y line
    x_range = np.linspace((16/15) * axes_limits[0], (372/400) * axes_limits[1])
    sns.lineplot(x=x_range, y=x_range, ls='--', color='k')
    ax.set_xlim(axes_limits)
    ax.set_ylim(axes_limits)

    # Add colorbar and legend
    labels(rf'Real Value of {length_variable} ($10^3$ km)', rf'Predicted {length_variable} ($10^3$ km)')
    add_flare_color_bar(ax)

    # Inset plot
    ax2 = ax.inset_axes([0.14, 0.73, 0.27, 0.25])
    flat_error, flat_error_delay, flat_hue, flat_predicted_change = [], [], [], []
    for i in range(len(error)):
        flat_error += error[i]
        flat_hue += error_year[i]
        flat_predicted_change += predicted_growth[i]
        for j in range(len(error[i])):
            flat_error_delay.append(i + 1)
    sns.scatterplot(x=flat_error_delay, y=flat_error, hue=flat_hue, ax=ax2, s=10, alpha=0.75, palette=c)

    # closed formula for linear regression with no intercept
    m = np.sum(np.array(flat_error_delay) * np.array(flat_error)) / np.sum(np.array(flat_error_delay)**2)
    x_axis = np.array(flat_error_delay)
    y_axis = m * x_axis
    ax2.plot(x_axis, y_axis, color='k', alpha=1, lw=0.7)
    ax2.get_legend().remove()

    # details
    ax2.set_xlabel(r'Delay (years)')
    ax2.set_ylabel('Absolute Error (%)')
    ax2.set_xticks([0, 5, 10, 15])
    # ax2.set_xticklabels([0, 50, 100])
    ax2.set_xlim(left=0)
    ax2.set_yticks([0.0, 0.07, 0.14, 0.21])
    ax2.set_yticklabels([0, 7, 14, 21])
    ax2.set_ylim(bottom=-0.005)

    # add years and R^2
    ax.text(*year_labels[0], '2008', rotation=90)
    ax.text(*year_labels[1], '2012', rotation=90)
    ax.text(*year_labels[2], '2016', rotation=90)
    ax.text(*year_labels[3], '2020', rotation=90)
    ax.text(200, 165, f'R$^2 = {r2:.2f}$')

    if save:
        save_figure(f'multiyear_prediction{name_suffix}')
    else:
        plt.show()

    return r2, m, max(flat_error)


def bootstrap_prediction(df, input_year, target_year, random_state, repetitions, length_variable):
    bootstrap = 0
    for i in range(repetitions):
        bootstrap += predict_year(df, input_year, target_year, random_state, length_variable)
    return bootstrap / repetitions


def predict_year(df, input_year, target_year, random_state, length_variable):

    # Using the data from the training year to find beta and A_i for cities
    input_df = df[df['Year'] == input_year]
    beta = exponential_regression_coefficients(input_df, ['Served Population'], [length_variable])
    beta = beta[0][0][0]
    input_df['Ai'] = input_df[length_variable] / (input_df['Served Population'] ** beta)

    # Splitting the target set between A (with sewer in 'year') and B (without sewer in 'year')
    target_df = df[df['Year'] == target_year]
    filter_for_A = input_df['City Code'].values.tolist()
    A = target_df[target_df['City Code'].isin(filter_for_A)]
    B = target_df[~target_df['City Code'].isin(filter_for_A)]

    # Estimating the sewer length for cities already in the dataset (_y = input, _x = target)
    A = pd.merge(A, input_df, on='City Code')
    A['Li'] = A['Ai'] * (A['Served Population_x']) ** beta
    La = A['Li'].sum()

    # Getting a random list with values of A_i and adding them to B
    pool = A['Ai'].sample(B.shape[0], replace=True, random_state=random_state)
    B['Ai'] = list(pool)

    # Estimating the sewer length for cities not in the dataset
    B['Li'] = B['Ai'] * B['Served Population'] ** beta
    Lb = B['Li'].sum()

    # Return the sum of length for cities in the data and outside the data
    final_length = La + Lb
    return final_length


def growth_paths(std, historical_data, data_path, beta, last_year, li_dict, load_data=True, save=False,
                 interpolation=1000, gini=False):
    # Historical data
    M, Y = [], []
    historical_mu, historical_std = [], []     # These will be used to plot the inequality later on
    for year in range(2005, last_year + 1):
        year_data = historical_data[historical_data['Year'] == year]
        historical_population = std[std['Year'] == year]
        M.append(year_data['Served Population'].sum())
        Y.append(year_data['Sewer Length'].sum())
        historical_mu.append(np.mean(historical_population['Attended Population'] / historical_population['Total Population']))
        historical_std.append(np.std(historical_population['Attended Population'] / historical_population['Total Population']))

    # Running the simulations
    std = std[std['Year'] == last_year]
    std['Li'] = std['City Code'].apply(lambda x: li_dict[x])
    std['Ai'] = std['Li'] / ((std['Total Population']) ** beta)
    std['Rate'] = (std['Ai'] * beta * std['Attended Population'] ** (beta - 1))
    std = std.sort_values(by='Rate').reset_index()
    efficiency_data = efficiency_growth(std, Y[-1], M[-1], gini)

    # Define the x_range as 1000 points in between first and last length values
    x_range = np.linspace(efficiency_data[0][0], efficiency_data[0][-1], interpolation) / 1e6

    if not load_data:
        efficiency_path, efficiency_gini, efficiency_mean, efficiency_std = growth_to_paths(efficiency_data, x_range)
        equality_path, equality_gini, equality_mean, equality_std = growth_to_paths(
            equality_growth(std, beta, Y[-1], M[-1], gini), x_range)

        with open(data_path, "wb") as file:
            pickle.dump((efficiency_path, equality_path, efficiency_gini, equality_gini,
                         efficiency_mean, equality_mean, efficiency_std, equality_std), file)

    with open(data_path, "rb") as fp:
        efficiency_path, equality_path, efficiency_gini, equality_gini, efficiency_mean, equality_mean, \
        efficiency_std, equality_std = pickle.load(fp)

    # Plotting the historical data
    new_plot()
    fig, ax1 = plt.subplots()
    ax1.plot(np.array(Y) / 1e3, np.array(M) / 1e6, c=(0.3, 0.3, 0.3))

    # Plotting
    ax1.plot(1000 * x_range, efficiency_path, c=ORANGE, ls='--', label='Optimized Efficiency')
    ax1.plot(1000 * x_range, equality_path, c=PURPLE, ls='--', label='Optimized Equality')

    # add points
    ax1.scatter(x=[np.array(Y)[0] / 1e3, np.array(Y)[-1] / 1e3],
                y=[np.array(M)[0] / 1e6, np.array(M)[-1] / 1e6], color=(0.3, 0.3, 0.3), s=10, zorder=10)
    ax1.text(162, 58.5, '2005', fontsize=10)
    ax1.text(330, 100, f'{last_year}', fontsize=10)

    # Additional styling
    ax1.legend(loc=4, frameon=False)
    labels('Sewer Length (10$^3$ km)', 'Population Served (10$^6$ inhabitants)')
    ax1.set_ylim(45, 260)
    ax1.set_xlim(90, 810)

    # Inset plot
    ax2 = ax1.inset_axes([0.14, 0.64, 0.35, 0.35])
    std['Priority'] = std.index
    std['log(Total Population)'] = np.log10(std['Total Population'])
    std['1/Ai'] = 1 / std['Ai']
    c = sns.color_palette("flare", as_cmap=True)

    # This is used to filter out the data points that have the Ai value sampled
    filtered_std = std[std['City Code'].isin(
        historical_data[historical_data['Year'] == last_year]['City Code'].to_list()
    )]

    sns.scatterplot(data=filtered_std.sample(frac=1, random_state=0), x='Priority', y='1/Ai',
                    hue='log(Total Population)', alpha=0.5, ax=ax2, s=18, palette=c)
    ax2.get_legend().remove()
    ax2.set_yscale('log')
    ax2.set_ylim(0.5e0, 0.6e5)

    sns.despine()
    inset_color_bar(ax1)

    if save:
        save_figure('brazil_growth')
    else:
        plt.show()

    return 1000 * x_range, efficiency_path, equality_path


def plot_colored_lines_with_interpolation(x, y, figure_ax, num_interp_points=100, colorbar=False):
    f = interp1d(x, y, kind='cubic')
    x = np.linspace(np.min(x), np.max(x), num_interp_points)
    y = f(x)
    z = y / np.sqrt(x * (1 - x))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True),
                        norm=Normalize(vmin=0, vmax=1))
    lc.set_array(z)
    lc.set_linewidth(3)
    figure_ax.add_collection(lc)
    figure_ax.autoscale()
    figure_ax.margins(0.1)

    # # Add a colorbar for the new plot
    # if colorbar:
    #     cb = plt.colorbar(lc, ax=ax, shrink=0.75)
    #     cb.outline.set_visible(False)
    #     cb.set_label('Infrastructure Inequality', fontsize=11)


def growth_to_paths(growth, x_range):
    length_sum, pop_sum, ginis, means, stds = growth
    length_sum, pop_sum, ginis, means, stds = \
        np.array(length_sum) / 1e6, np.array(pop_sum) / 1e6, np.array(ginis), np.array(means), np.array(stds)
    path = np.interp(x_range, length_sum, pop_sum)
    gini = np.interp(x_range, length_sum, ginis)
    mean = np.interp(x_range, length_sum, means)
    std = np.interp(x_range, length_sum, stds)
    return path, gini, mean, std


def efficiency_growth(df, initial_length, initial_pop, gini):
    # Calculate the most efficient cities to grow and sort by efficiency
    df['Rate'] = (df['Total Population'] - df['Served Population']) / (df['Li'] - df['Sewer Length'])
    df = df.sort_values(by=['Rate'], ascending=False).reset_index()
    df['temp_k'] = df['Served Population'] / df['Total Population']

    length_sum = [initial_length]
    pop_sum = [initial_pop]
    ginis = [infrastructure_inequality(df['temp_k'].tolist(), gini)]
    means = [np.mean(df['temp_k'].tolist())]
    stds = [np.std(df['temp_k'].tolist())]

    boundary = 1

    while boundary <= df.shape[0]:
        optimized = df[:boundary]
        df.loc[boundary - 1, 'temp_k'] = 1
        length_sum.append(optimized['Li'].sum() - optimized['Sewer Length'].sum() + initial_length)
        pop_sum.append(optimized['Total Population'].sum() - optimized['Served Population'].sum() + initial_pop)
        ginis.append(infrastructure_inequality(df['temp_k'].tolist(), gini))
        means.append(np.mean(df['temp_k'].tolist()))
        stds.append(np.std(df['temp_k'].tolist()))
        boundary += 1

    return length_sum, pop_sum, ginis, means, stds


def equality_growth(df, beta, initial_length, initial_pop, gini, step_size=1):
    length_sum = [initial_length]
    pop_sum = [initial_pop]
    df['temp_Y'] = df['Sewer Length']
    df['temp_M'] = df['Served Population']
    df['temp_k'] = df['temp_M'] / df['Total Population']
    df = df.sort_values(by=['temp_k'], ascending=True).reset_index()
    ginis = [infrastructure_inequality(df['temp_k'].tolist(), gini)]
    means = [np.mean(df['temp_k'].tolist())]
    stds = [np.std(df['temp_k'].tolist())]
    min_temp_k_index = 0

    while df.loc[min_temp_k_index, 'temp_k'] < 1:

        # Find the correct value to increase the sewers by
        delta_Y = step_size  # , df.loc[min_temp_k_index, 'Li'] - df.loc[min_temp_k_index, 'temp_Y'])
        df.loc[min_temp_k_index, 'temp_Y'] = df.loc[min_temp_k_index, 'temp_Y'] + delta_Y

        # Attending more people in the city and seeing effects on Y and k
        delta_M = (df.loc[min_temp_k_index, 'temp_Y'] / df.loc[min_temp_k_index, 'Ai']) ** (1 / beta) \
                  - df.loc[min_temp_k_index, 'temp_M']
        df.loc[min_temp_k_index, 'temp_M'] = min(df.loc[min_temp_k_index, 'temp_M'] + delta_M,
                                                 df.loc[min_temp_k_index, 'Total Population'])
        df['temp_k'] = df['temp_M'] / df['Total Population']

        # Updating the current values for dM, dY, and ginis
        pop_sum.append(pop_sum[-1] + delta_M)
        length_sum.append(length_sum[-1] + delta_Y)
        ginis.append(infrastructure_inequality(df['temp_k'].tolist(), gini))
        means.append(np.mean(df['temp_k'].tolist()))
        stds.append(np.std(df['temp_k'].tolist()))

        # Find the city to be expanded next
        min_temp_k_index = df['temp_k'].idxmin()

    return length_sum, pop_sum, ginis, means, stds


def gini_coefficient(values):
    values = np.array(values)
    n = len(values)
    mean_value = values.mean()

    # Using broadcasting to compute the absolute differences
    abs_diffs = np.abs(values[:, None] - values).sum()
    gini = abs_diffs / (2 * n ** 2 * mean_value)
    return gini


# Formula from the Infrastructure inequality is a characteristic of urbanization' paper
def infrastructure_inequality(values, gini=False):
    if gini:
        return gini_coefficient(values)
    avg = np.mean(values)
    std = np.std(values)
    if avg <= 0 or avg >= 1:
        return 0
    else:
        return std / np.sqrt(avg * (1 - avg))


def inset_color_bar(ax, title='logN'):
    norm = plt.Normalize(2.892651, 7.088208)
    s_map = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    s_map.set_array(np.array([]))
    color_bar = inset_axes(ax,
                           width="30%",
                           height="1.5%",
                           loc='upper left',
                           bbox_to_anchor=(0.20, -0.24, 0.68, 1.18),
                           bbox_transform=ax.transAxes)
    try:
        ax.get_legend().remove()
    except:
        pass
    cb = ax.figure.colorbar(s_map, cax=color_bar, ticks=[3.5, 5.0, 6.5], orientation='horizontal')
    cb.outline.set_visible(False)
    cb.ax.set_title(title, fontsize=11)


def simulate_full_bootstrap(df, full, growth, growth_size=227638581, seed=0, n=100, show_distribution=False):

    full_lengths = []

    results = {cc: 0 for cc in full['City Code'].to_list()}
    for seed_shift in tqdm(range(n)):
        A, B = simulate_full_sample(df, full, seed + seed_shift)
        for dataset in [A, B]:
            for i in range(dataset.shape[0]):
                code, Li = dataset.iloc[i, 0], dataset.iloc[i, 1]
                results[code] += Li
        full_lengths.append(A['Li'].sum() + B['Li'].sum())

    if growth:
        full_population = full['Total Population'].sum()
        g = (growth_size - full_population) / full_population
        print('Growth:', g)

        beta = log_linear_regression(df, ['Served Population'], ['Sewer Length'])[0][0][0]
        multiplier = ((1 + g) ** beta)
    else:
        multiplier = 1
    full['Li'] = full['City Code'].apply(lambda x: multiplier * results[x] / n)

    # Get the 95% confidence interval
    lower_bound = np.percentile(full_lengths, 2.5)
    upper_bound = np.percentile(full_lengths, 97.5)

    if show_distribution:
        # Plot the distribution of results
        sns.set_style('white')
        sns.histplot(np.array(full_lengths) / 1000)
        plt.xlabel(rf'Predicted Sewer Length ($10^3$ km)')
        sns.despine()
        plt.show()

    return full, np.average(full_lengths), lower_bound, upper_bound


# Simulated the total length necessary to achieve full access in the country
def simulate_full_sample(df, full, seed):

    # Using the data from the training year to find beta and A_i for cities
    beta = log_linear_regression(df, ['Served Population'], ['Sewer Length'])[0][0][0]
    df['Ai'] = df['Sewer Length'] / (df['Served Population'] ** beta)

    # Splitting the set between A (with water in 'year') and B (without water in 'year')
    filter_for_A = df['City Code'].values.tolist()
    A = full[full['City Code'].isin(filter_for_A)]
    B = full[~full['City Code'].isin(filter_for_A)]

    # Estimating the sewer length for cities already in the dataset (_y = df, _x = full)
    A = pd.merge(A, df, on='City Code')
    A['Li'] = A['Ai'] * (A['Total Population_x']) ** beta

    # Getting a random list with values of A_i and adding them to B
    pool = A['Ai'].sample(B.shape[0], replace=True, random_state=seed)
    B['Ai'] = list(pool)

    # Estimating the sewer length for cities not in the dataset
    B['Li'] = B['Ai'] * B['Total Population'] ** beta

    A = A[['City Code', 'Li']].reset_index(drop=True)
    B = B[['City Code', 'Li']].reset_index(drop=True)

    return A, B

