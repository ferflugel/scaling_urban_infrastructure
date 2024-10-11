import pandas as pd

def load_snis_data(file):

    # Overall settings for the data being collected
    df = pd.read_csv(file)
    df = df[df['Year'] >= FIRST_SNIS_YEAR]

    # Clean data for sewers
    sewer = df.dropna(subset=['Sewer Length', 'Attended Population (sewer)'])
    sewer = sewer[sewer['Sewer Length'] > 1]
    sewer = sewer[sewer['Attended Population (sewer)'] > 100]
    sewer = sewer[sewer['Sewer Volume'] > 1]
    sewer['Percent Served'] = sewer['Attended Population (sewer)'] / sewer['Total Population']
    sewer['Served Population'] = sewer['Attended Population (sewer)']

    # Clean data for water
    water = df.dropna(subset=['Water Length', 'Attended Population (water)'])
    water = water[water['Water Length'] > 1]
    water = water[water['Attended Population (water)'] > 100]
    water = water[water['Water Volume Produced'] > 1]
    water = water[water['Water Volume Consumed'] > 1]
    water['Percent Served'] = sewer['Attended Population (water)'] / sewer['Total Population']
    water = water[water['Percent Served'] <= 1]
    water['Served Population'] = water['Attended Population (water)']

    # Deleting bad data entries
    sewer = sewer[(sewer['City Code'] != 330060) | (sewer['Year'] != 2011)]
    sewer = sewer[(sewer['City Code'] != 411270) | (sewer['Year'] != 2011)]
    sewer = sewer[(sewer['City Code'] != 432225) | (sewer['Year'] != 2017)]
    sewer = sewer[(sewer['City Code'] != 320410) | (sewer['Year'] != 2019)]

    return sewer, water


def load_ibnet_data(file, year):

    # Load the IBNET dataset
    df = pd.read_excel(file)                 # Load the raw IBNET data
    df = df[df['Year'] == year].copy()       # Filter the data only for the specified year

    # Exclude Brazilian networks
    df = df[df['Country'] != 'Brazil'].copy()

    # Selecting the columns and defining their new names
    sewer_columns = ['City', 'Region', 'Country', 'Code', 'Year', 'R_30A_TOTAL_POP_WASTE',
                     'R_70_POP_SERVED_SEWERAGE', 'R_74_LENGTH_SEWERS']

    water_columns = ['City', 'Region', 'Country', 'Code', 'Year', 'R_30_TOTAL_POP_WATER_SUPPLY',
                     'R_40_POP_SERVED_WATER', 'R_54_LENGTH_WATER_DIST_NETWORK']

    sewer_labels = {'R_30A_TOTAL_POP_WASTE': 'Total Population',
                    'R_70_POP_SERVED_SEWERAGE': 'Served Population',
                    'R_74_LENGTH_SEWERS': 'Sewer Length'}

    water_labels = {'R_30_TOTAL_POP_WATER_SUPPLY': 'Total Population',
                    'R_40_POP_SERVED_WATER': 'Served Population',
                    'R_54_LENGTH_WATER_DIST_NETWORK': 'Water Length'}

    # Filter and rename columns; drop rows with null entries
    sewer = df[sewer_columns].copy()
    sewer = sewer.dropna()
    sewer = sewer.rename(columns=sewer_labels)

    # Clean the data for sewers
    sewer['Percent Served'] = sewer['Served Population'] / sewer['Total Population']
    sewer = sewer[sewer['Sewer Length'] > 1]
    sewer = sewer[sewer['Served Population'] > 100]
    sewer = sewer[sewer['Percent Served'] <= 1]

    # Filter and rename columns; drop rows with null entries
    water = df[water_columns].copy()
    water = water.dropna()
    water = water.rename(columns=water_labels)

    # Clean the data for water
    water['Percent Served'] = water['Served Population'] / water['Total Population']
    water = water[water['Water Length'] > 1]
    water = water[water['Served Population'] > 100]
    water = water[water['Percent Served'] <= 1]

    # Exceptions
    water = water[water['Code'] != 'BH84']      # Length is not reasonable

    return sewer, water


if __name__ == '__main__':

    # Settings for the data loading process
    FIRST_SNIS_YEAR = 2005
    IBNET_YEAR = 2015

    # Run the function that cleans the SNIS data and save it
    snis_sewer, snis_water = load_snis_data('raw/SNIS_2020_raw.csv')
    snis_sewer.to_csv('clean/snis_sewer.csv', index=False)
    snis_water.to_csv('clean/snis_water.csv', index=False)

    # Run the function that cleans the IBNET data and save it
    ibnet_sewer, ibnet_water = load_ibnet_data('raw/IBNET_raw.xlsx', IBNET_YEAR)
    ibnet_sewer.to_csv('clean/ibnet_sewer.csv', index=False)
    ibnet_water.to_csv('clean/ibnet_water.csv', index=False)
