import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_data(filename, separator=""):
    """
    Read data from a CSV file.

    Parameters:
        filename (str): The name of the CSV file.
        separator (str): The delimiter used in the CSV file. Default is an empty string.

    Returns:
        DataFrame: The DataFrame containing the data from the CSV file.
    """

    try:
        if not separator:
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename, sep=separator, engine="python", encoding="utf-8")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV file '{filename}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def describe_dataframe(data):
    """
    Generates a textual summary of the dataset including its shape, column names, data types,
    total missing values, and basic statistics.

    Args:
        data (DataFrame): The dataset to describe.

    Returns:
        str: Textual summary of the dataset.
    """
    description = ""
    spacer = "-" * 30
    description += f"ROWS and COLUMNS \n{spacer}\n{data.shape}\n\n"
    description += f"COLUMN NAMES \n{spacer}\n{data.columns.tolist()}\n\n"
    description += f"DATA TYPES \n{spacer}\n{data.dtypes}\n\n"
    description += f"TOTAL MISSING VALUES \n{spacer}\n{data.isnull().sum()}\n\n"
    description += f"STATISTICS \n{spacer}\n{data.describe()}"

    return description


def count_categorical(data):

    """
    Counts the occurrences of each category in the categorical columns of a DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary where the keys are the column names and the values are the counts of unique values in each categorical column.
    """
    categorical_data_counts = {}

    df = data.select_dtypes(include=["object"]) 

    categorical_columns = df.columns

    for column in categorical_columns:
        categorical_data_counts[column] = df[column].value_counts()
    
    return categorical_data_counts


def plot_categorical_grouped_boxplots(data, column_list):
    """
    Plots a grouped boxplot for categorical data.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    column_list (list): List of column names to be included in the plot.
                        Must include exactly one categorical column and at least one numerical column.

    Returns:
    None
    """
    df = data[column_list]

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Check if there is exactly one categorical column
    if len(categorical_columns) != 1:
        raise ValueError("The column_list must contain exactly one categorical column.")
    else:
        categorical_column = categorical_columns[0]

    # Check if there is at least one numerical column
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_columns:
        raise ValueError("The column_list must contain at least one numerical column.")

    plot_data = (df
                 .set_index(categorical_column)
                 .stack()
                 .to_frame()
                 .reset_index()
                 .rename(columns={0: 'Feature Values', 'level_1': 'Features'})
                 )

    sns.boxplot(data=plot_data, x="Features", y="Feature Values", hue=categorical_column, palette="pastel")
    plt.show()


def plot_pairplot(data, column_list):
    """
    Plots a pairplot for visualizing relationships between numerical columns, with hue specified by a categorical column.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    column_list (list): List of column names to be included in the plot.
                        Must include exactly one categorical column and at least one numerical column.

    Returns:
    None
    """
    df = data[column_list]

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Check if there is exactly one categorical column
    if len(categorical_columns) != 1:
        raise ValueError("The column_list must contain exactly one categorical column.")
    else:
        categorical_column = categorical_columns[0]

    # Check if there is at least one numerical column
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_columns:
        raise ValueError("The column_list must contain at least one numerical column.")

    sns.set_context('talk')
    sns.pairplot(df, hue=categorical_column, palette="pastel")
    plt.show()


def plot_numerical_distributions(data, save_path):
    """
    Plot boxplots to visualize the distributions of numerical features.

    Parameters:
        data (DataFrame): The DataFrame containing numerical features.
        save_path (str): The file path to save the plot.
    """

    df = data.select_dtypes(include=["float64", "int64"])
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
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Distribution")

    # If there are any leftover axes, hide them
    for j in range(num_columns, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_categorical(data, save_path):
    """
    Create pie charts to visualize the distribution of categorical data in each column.

    Parameters:
        data (DataFrame): The DataFrame containing categorical data.
        save_path (str): The file path to save the plot.
    """
    # Select only categorical columns
    categorical_columns = data.select_dtypes(include=["object"])

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
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(30, 6 * num_rows),
        subplot_kw={"aspect": "equal"},
    )

    # Flatten axes array for easy iteration if it's multidimensional
    if num_plots != 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]  # Single subplot case

    # Define color palettes for each pie chart
    color_palettes = ["Pastel1"]

    # Iterate over each column in the big dictionary
    for i, (column_name, (string_list, frequency_list)) in enumerate(
        categorical_stats_dict.items()
    ):
        # Choose color palette for current pie chart
        color_palette = plt.get_cmap(color_palettes[i % len(color_palettes)])

        # Create pie chart for the current column
        patches, texts, autotexts = axes_flat[i].pie(
            frequency_list,
            labels=string_list,
            autopct="%1.1f%%",
            startangle=90,
            colors=color_palette.colors,
        )

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
        legend_labels = [
            f"{label}: {freq} ({100 * freq / sum(frequency_list):.1f}%)"
            for label, freq in zip(string_list, frequency_list)
        ]
        axes_flat[i].legend(
            patches,
            legend_labels,
            title=column_name,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

    # Hide any unused subplot axes
    for j in range(num_plots, len(axes_flat)):
        axes_flat[j].axis("off")

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)

    plt.close()


def plot_correlation_matrix(data, save_path, threshold=False):
    """
    Plot a heatmap of the correlation matrix for numerical data.

    Parameters:
        data (DataFrame): The DataFrame containing numerical data.
        save_path (str): The file path to save the plot.
        threshold (float or False): Threshold value for removing low correlations. If False, no thresholding is applied.
    """

    numerical_data = data.select_dtypes(include=["float64", "int64"])
    corr = numerical_data.corr()

    if threshold:
        # Apply thresholding to remove low correlations
        corr[(abs(corr) > 0) & (abs(corr) < threshold)] = 0
        corr[(abs(corr) > -threshold) & (abs(corr) < 0)] = 0

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-0.8,
        vmax=0.8,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)

    plt.close()


def plot_feature_correlation(dataframe, target_variable, save_path):
    """
    Plot pair scatter plots between numerical features and the target variable.

    Parameters:
        dataframe (DataFrame): The DataFrame containing numerical features and the target variable.
        target_variable (str): The name of the target variable.
        save_path (str): The file path to save the plot.
    """

    # Select numerical columns excluding the target variable
    numerical_columns = dataframe.select_dtypes(include=["float64", "int64"]).columns
    numerical_columns = numerical_columns.drop(target_variable)

    # Set the number of plots per row
    plots_per_row = 4

    # Calculate the number of rows needed
    num_plots = len(numerical_columns)
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(4 * plots_per_row, 4 * num_rows))
    axes = axes.flatten()

    # Iterate over numerical columns and create pair plots
    for i, column in enumerate(numerical_columns):
        sns.scatterplot(data=dataframe, x=column, y=target_variable, ax=axes[i])

        # Set plot title
        axes[i].set_title(f"{column} vs {target_variable}")

    # Hide any unused subplot axes
    for j in range(num_plots, len(axes)):
        axes[j].axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)

    plt.close()


def plot_normality(data, target_variable, save_path):
    """
    Visualizes the distribution of numerical variables in the given dataset and checks for normality.

    Args:
        data (DataFrame): The dataset containing the numerical variables.
        target_variable (str): The name of the target variable.
        save_path (str): The file path where the plot will be saved.

    Returns:
        None
    """
    df = data.select_dtypes(include=["float64", "int64"])
    num_plots = len(df.columns)
    num_cols = 4  # Number of plots per row

    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    for i, colname in enumerate(df.columns):
        ax = axes[i]

        sns.histplot(data=df[colname], color="#3ac5fe", kde=False, stat="density", ax=ax)
        sns.kdeplot(data=df[colname], color="#F76693", ax=ax)

        # Calculate skewness of the target variable
        skewness_value = round(data[colname].skew(), 2)

        # Determine skewness level
        textbox_text = ""
        if -0.5 < skewness_value < 0.5:
            textbox_text = "Symmetrical"
        elif (-1.0 < skewness_value < -0.5) or (0.5 < skewness_value < 1.0):
            textbox_text = "Moderate"
        else:
            textbox_text = "Highly"

        # Add textbox annotation for skewness level
        ax.text(
            0.95,
            0.95,
            f"{textbox_text}\nSkewness: {skewness_value}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="lightgray", alpha=0.5),
        )

    # Hide any unused subplot axes
    for j in range(num_plots, len(axes)):
        axes[j].axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)

    plt.close()


def plot_missing_values(data, data_filename):
    """
    Plots the total count of missing values for each column in the dataset.

    Args:
        data (DataFrame): The dataset containing missing values.
        data_filename (str): The name of the data file.

    Returns:
        None
    """
    filename_prefix = data_filename.split(".")[0]
    plot_filename = "missing_values_stats.png"
    save_path = os.path.join("data_statistics", filename_prefix, plot_filename)

    total = data.isnull().sum().sort_values(ascending=False)

    total[total > 1].plot(kind="bar", figsize=(8, 6), color="#3ac5fe", fontsize=10)

    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.title("Total Missing Values", fontsize=20)

    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)

    plt.close()


def data_analysis_workflow(data_file_name, target_variable=False, sep=False):
    """
    Perform a machine learning workflow including data loading, statistics plotting, and correlation analysis.

    Parameters:
        data_file_name (str): The file name of the data file.
        target_variable (str or False): The name of the target variable. Default is False.
        sep (str or False): The separator used in the data file. Default is False.
    """

    root_foldername = "data_statistics"
    data_file_prefix = data_file_name.split(".")[0]
    foldername = os.path.join(root_foldername, data_file_prefix)

    # Check if the folder exists
    if not os.path.exists(root_foldername):
        # Create the folder if it doesn't exist
        os.makedirs(root_foldername)

    # Check if the subfolder exists
    if not os.path.exists(foldername):
        # Create the folder if it doesn't exist
        os.makedirs(foldername)

    if not sep:
        data = read_data(data_file_name)
    else:
        data = read_data(data_file_name, sep)

    num_stats_filepath = os.path.join(foldername, "Numerical_Data_Stats.png")
    cat_stats_filepath = os.path.join(foldername, "Categorical_Data_Stats.png")
    corr_stats_filepath = os.path.join(foldername, "Correlation_Matrix_Stats.png")
    corr_feature_stats_filepath = os.path.join(foldername, "Correlation_Features_Stats.png")
    norm_stats_filepath = os.path.join(foldername, "Normal_Distribution_Stats.png")

    # Plotting numerical data distributions
    plot_numerical_distributions(data, num_stats_filepath)

    # Plotting cagegorical columns data statistics
    plot_categorical(data, cat_stats_filepath)

    # Plotting correlation matrix
    plot_correlation_matrix(data, corr_stats_filepath, threshold=0.5)

    if target_variable:
        # Plotting feature correlation
        plot_feature_correlation(data, target_variable, corr_feature_stats_filepath)

    plot_normality(data, target_variable, norm_stats_filepath)

    plot_missing_values(data, data_file_name)
