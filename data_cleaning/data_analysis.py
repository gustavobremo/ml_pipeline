import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


# Reading data
def read_data(filename, separator=""):
    try:
        if not separator:
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename, sep=separator, engine='python', encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV file '{filename}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# View data table content information
def overview(df, head=5):
    """
    Provides an overview of a DataFrame including its dimensions, the first few rows,
    and the data types of its columns.

    Parameters:
        df (DataFrame): The DataFrame to be summarized.
        head (int): Number of rows to display.

    Returns:
        info (str): Summary information about the DataFrame.
    """

    if head < 0:
        print("Error: head argument must be non-negative.")
        return None

    # dim = df.shape
    # info = f"\nTotal of {dim[0]} rows and {dim[1]} columns in dataset:\n\n"
    info = f"\nFirst {head} rows of the dataframe:\n"
    info += f"{df.head(head)}\n\n"
    info += "Column data types summary:\n"

    df_info = df.info(verbose=False, memory_usage='deep')
    
    # Concatenate info if it's not None
    print(info)
    print(df.info())



def plot_numerical_distributions(data, save_path):
    df = data.select_dtypes(include=['float64', 'int64'])
    num_columns = df.shape[1]  # Get number of columns in the DataFrame
    num_rows = (num_columns + 3) // 4  # Calculate the required number of rows (4 columns per row)

    # Create a figure with subplots in a grid with 4 columns per row
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 5 * num_rows))
    axes = axes.flatten()  # Flatten the array of axes to simplify indexing

    # Get a color from the palette
    colors = sns.color_palette("Pastel1")

    # Iterate over each column and create a distribution plot
    for i, column in enumerate(df.columns):
        sns.boxplot(y=df[column], ax=axes[i], color=colors[0])
        axes[i].set_title(column)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Distribution')

    # If there are any leftover axes, hide them
    for j in range(num_columns, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path)


def plot_categorical(data, save_path):
    """
    Create pie charts to visualize the distribution of categorical data in each column.

    Parameters:
        data (DataFrame): The DataFrame containing categorical data.
        save_path (str): The file path to save the plot.
    """
    # Select only categorical columns
    categorical_columns = data.select_dtypes(include=['object'])

    # Create a dictionary with value counts for each column
    description_dict = {col: data[col].value_counts() for col in categorical_columns}

    # Filter columns with fewer than 10 unique categories
    categorical_stats_dict = {
        col: (list(counts.index), list(counts.values))
        for col, counts in description_dict.items()
        if len(counts) <= 10
    }

    num_plots = len(categorical_stats_dict)
    num_rows = int(np.ceil(num_plots / 2))  # Calculate number of rows needed for 2 plots per row
    num_cols = 2  # Set number of columns to 2

    # Specify the overall size of the figure (width, height in inches)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 6 * num_rows), subplot_kw={'aspect': 'equal'})

    # Flatten axes array for easy iteration if it's multidimensional
    if num_plots != 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]  # Single subplot case

    # Define color palettes for each pie chart
    color_palettes = ['Pastel1']

    # Iterate over each column in the big dictionary
    for i, (column_name, (string_list, frequency_list)) in enumerate(categorical_stats_dict.items()):
        # Choose color palette for current pie chart
        color_palette = plt.get_cmap(color_palettes[i % len(color_palettes)])

        # Create pie chart for the current column
        patches, texts, autotexts = axes_flat[i].pie(frequency_list, labels=string_list, autopct='%1.1f%%', startangle=90, colors=color_palette.colors)

        # Customize text size for legibility
        for text in texts:
            # text.set_size('large')
            text.set_visible(False)
        for autotext in autotexts:
            autotext.set_visible(False)

        # Set plot title
        axes_flat[i].set_title(column_name)

        # Configure legend
        # Prepare legend handles and labels by extracting data from patches and string_list
        legend_labels = [f'{label}: {freq} ({100 * freq / sum(frequency_list):.1f}%)' for label, freq in zip(string_list, frequency_list)]
        axes_flat[i].legend(patches, legend_labels, title=column_name, loc='upper left', bbox_to_anchor=(1, 1))

    # Hide any unused subplot axes
    for j in range(num_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)



def plot_correlations(data, response_var, save_path):
    df = data.select_dtypes(include=['float64', 'int64'])

    hous_num = df.select_dtypes(include=['float64', 'int64'])
    hous_num_corr = hous_num.corr()[response_var][:-1]  # -1 means that the latest row is SalePrice
    top_features = hous_num_corr[abs(hous_num_corr) > 0.5].sort_values(ascending=False)  # Displays Pearson's correlation coefficient greater than 0.5
    print("There are {} strongly correlated values with:\n{}".format(len(top_features), top_features))

    num_columns = len(hous_num.columns)
    num_rows = (num_columns + 3) // 4  # Calculate the required number of rows (4 plots per row)

    # Create a figure with subplots in a grid with 4 plots per row
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 5 * num_rows))

    # Iterate over each subset of columns and create pair plots
    for i in range(0, num_columns, 4):
        subset_columns = hous_num.columns[i:i+4]
        g = sns.PairGrid(data=hous_num, x_vars=subset_columns, y_vars=[response_var])
        g.map(sns.scatterplot)
        # Set the axes for the pair plot
        for row in range(num_rows):
            for col in range(4):
                if i + col < num_columns:
                    g.axes[row, col] = axes[row, col + i]

    # Hide any excess axes
    for j in range(num_columns, num_rows * 4):
        fig.delaxes(axes.flatten()[j])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)


def main():
    data = read_data("Ames_Housing_Data1.tsv","\t")
    # data = read_data("supermarket_sales.csv")     

    # plot_correlations(data,"SalePrice", "A.png")
    # plot_correlations(data,"Unit price")

    plot_numerical_distributions(data,"Numerical_Data_Stats.png")

    # Plotting cagegorical columns data statistics
    plot_categorical(data,"Categorical_Data_Stats.png")
    
main()