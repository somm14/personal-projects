import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.subplots as sp
from scipy.stats import chi2_contingency
import seaborn as sns



def describe_df(df):
    '''
    This function displays specific information about the original DataFrame (DF).
    The information includes: the object type, the % of missing values,
    unique values, and the % of cardinality for each column in the original DF.

    Arguments:
    df (pd.DataFrame): The original DF for which we wan to retrieve the information.

    Returns:
    pd.DataFrame: A DF with the specific information.
    '''
    # Create a dictionary with the column that will stay fixed
    # Then add the columns from the original DF
    dict_col = {'COL_N': ['DATA_TYPE', 'MISSINGS (%)', 'UNIQUE_VALUES', 'CARDIN (%)']}

    # Formula to calculate the % of missing values
    na_ratio = ((df.isnull().sum() / len(df))*100)

    # Add the column names as keys and their descriptive information as values.
    for col in df:
        dict_col[col] = [df.dtypes[col], na_ratio[col], len(df[col].unique()), round(df[col].nunique()/len(df)*100,2)]

    # Create the DF.describe
    df_describe = pd.DataFrame(dict_col)

    return df_describe



def categorize_variables(df:pd.DataFrame, category_threshold:int, continuous_threshold:float):
    '''
    This function is used to categorize the variables, of a given DF into categorical, continuous numerical or discrete numerical.

    Arguments:
    df (pd.DataFrame): Original DF to acquire the variables to be categorized.
    category_threshold (int): An integer value representing the threshold to assign a variable as categorical.
    continuous_threshold (float): A float value representing the threshold to assign a variable as numerical.

    Returns:
    pd.DataFrame: A DF with two columns: 'variable_name' and 'suggested_type', containing as many rows as there are columns in the orginal DF.
    '''
    
    # Error handling
    if type(category_threshold) != int:
        raise TypeError(f'The value of "category_threshold" must be type {int}, but received {type(category_threshold)}')

    elif type(continuous_threshold) != float:
        raise TypeError(f'The value of "continuous_threshold" must be type {float}, but received {type(continuous_threshold)}')
    
    elif not isinstance(df, pd.DataFrame):
        raise TypeError(f'The input "df" must be a pandas DF, but receive {type(df)}')

    # Create categorization DF
    else:
        df_categorization = pd.DataFrame({
            'variable_name': df.columns
        })
        df_categorization['suggested_type'] = ''

        for i, val in df_categorization['variable_name'].items():
            card = df[val].nunique()
            porcentage = df[val].nunique()/len(df) * 100

            if card == 2:
                df_categorization.at[i,'suggested_type'] = 'Binary'
            
            elif card < category_threshold:
                df_categorization.at[i,'suggested_type'] = 'Categorical'
        
            else:
                if porcentage > continuous_threshold:
                    df_categorization.at[i, 'suggested_type'] = 'Continuous numerical'
                else:
                    df_categorization.at[i, 'suggested_type'] = 'Discrete numerical'
    
    return df_categorization



def list_categories(df_categorization, category_types=[]):
    '''
    Instantiation of dictionaries for each categorization type of the variables in a DF.

    Arguments:
    df_categorization (pd.DataFrame): DF containing the categorization information of each variable, with columns named 'variable_name' and 'suggested_type'. The first refers to the variable name, and the
    second to the categorization type.
    category_types (list[str]): List of categorization type names to create as dictionaries.

    Returns:
    result (dict): Creates lists of variables based on the given categorizarion type.
    '''
    result = {}
    for t in category_types:
        var_list = [var for var, cat in zip(df_categorization['variable_name'], df_categorization['suggested_type']) if cat == t]
        result[t] = var_list
    return result



def plot_cateforical_distribution_plotly(df, categorical_columns, relative=False, show_values=False):
    '''
    This function creates bar charts to visualize the distribution of categorical variables using Plotly.

    Arguments:
    df (pd.DataFrame): The DF containing the categorical variables.
    categorical_columns (list): List of categorical column names to be visualized.
    relative (bool): If True, displays relative frequencies instead of absolute counts. By default, False.
    show_values (bool): If True, displays values on the bars. By default, False.

    Returns:
    None: Displays a subplot figure with bar charts.
    '''
    num_columns = len(categorical_columns)
    num_rows = (num_columns // 2) + (num_columns % 2)

    # Create a figure with subplots
    fig = sp.make_subplots(rows=num_rows, cols=2, subplot_titles=categorical_columns)

    for i, col in enumerate(categorical_columns):
        # Compute absolute or relative frequencies
        series = df[col].value_counts(normalize=relative).reset_index()
        series.columns = [col, 'count']

        # Create bar chart
        bar_fig = px.bar(series, x=col, y='count', text_auto='.2f' if show_values else None)

        # Add chart to the main figure
        row = (i // 2) + 1
        col_pos = (i % 2) + 1
        for trace in bar_fig.data:
            fig.add_trace(trace, row=row, col=col_pos)
        
    # Final adjustments
    fig.update_layout(
        title_text="Categorical Variable Distribution",
        height=300 * num_rows,
        showlegend=False
    )

    fig.show()




def plot_categorical_relationship(df_data, categorical_column, target):
    '''
    This function visualizes the relationship between a categorical variable and the target variable using a stacked bar chart. It groups the data by the categorical variable and target, then plots the counts
    of clients with and without churn for each category

    Arguments:
    df_data (pd.DataFrame): The DF containing the data
    categorical_column (str): The name of the categorical column analyze.
    target (str): The name of the target variable.

    Returns:
    None: Displays a stacked bar chat showing the distribution of the target variable within each category of the specified categorical column.
    '''
    # Count target by category
    data_counts = df_data.groupby([categorical_column, target]).size().reset_index(name="Clients")

    # Stacked bar chart
    fig = px.bar(data_counts, x=categorical_column, y="Clients", color=target,
                title=f"Distribution of {target} by {categorical_column}", barmode="stack")

    fig.show()



def plotly_grouped_histograms(df_data, cat_col, num_col, group_size=3):
    '''
    This function creates grouped histograms to visualize the distribution of a numerical variable gruoped by a categorical variable.

    Arguments:
    df_data (pd.DataFrame): The DF containing the data
    cat_col (str): The name of the categorical column to group by.
    num_col (str): The name of the numerical column to be visualized.
    group_size (int): The number of categories to gropu in each histogram plot. Default is 3.

    Returns:
    None: Displays interactive histograms grouped by the specified categorical variable.
    '''
    unique_cats = df_data[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i + group_size]
        subset_df_data = df_data[df_data[cat_col].isin(subset_cats)]
        
        # List of data for each category
        hist_data = [subset_df_data[subset_df_data[cat_col] == cat][num_col] for cat in subset_cats]
        
        # Create legend labels
        group_labels = [f"{cat_col}: {cat}" for cat in subset_cats]

        # Create distribution plot using Plotly
        fig = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=True)
        
        # Final adjustments
        fig.update_layout(
            title=f'Histogram of {num_col} por {cat_col} (Group {i//group_size + 1})',
            xaxis_title=num_col,
            yaxis_title="Density",
            legend_title=cat_col
        )

        fig.show()



def plot_combined_graphs(df_data, columns, whisker_width=1.5, bins = None):
    '''
    This function creates histograms with KDE and boxplots for numerical columns using Matplotlib and Seaborn.

    Arguments:
    df_data_data (pd.DataFrame): The DataFrame containing numerical columns.
    columns (list): List of numerical column names to be visualized.
    whisker_width (float): The whisker width for the boxplot. Default is 1.5.
    bins (str or int): Number of bins for the histogram. Default is None.

    Returns:
    None: Displays histograms and boxplots for the selected numerical columns.
    '''
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df_data[column].dtype in ['int64', 'float64']:
                # Histogram and KDE
                sns.histplot(df_data[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histogram and KDE for {column}')
                else:
                    axes[0].set_title(f'Histogram and KDE for {column}')

                # Boxplot
                sns.boxplot(x=df_data[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot for {column}')
                else:
                    axes[1].set_title(f'Boxplot for {column}')

        plt.tight_layout()
        plt.show()



def chi_square_matrix(df):
    """
    Computes the Chi-Square statistic for all pairs of categorical variables in a given DataFrame and returns a symmetric matrix of Chi-Square values.

    Arguments:
    df (pd.DataFrame): The DataFrame containing categorical variables.

    Returns:
    pd.DataFrame: A symmetric matrix where each entry represents the Chi-Square statistic for the corresponding variable pair.
    """
    chi2_matrix = np.zeros((df.shape[1], df.shape[1]))

    for i in range(df.shape[1]):
        for j in range(i, df.shape[1]):
            # Create the contingency table
            contingency_table = pd.crosstab(df.iloc[:, i], df.iloc[:, j])
            # perform the chi-square test
            chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
            chi2_matrix[i, j] = chi2_stat
            chi2_matrix[j, i] = chi2_stat  # the matrix is ​​symmetric

    return pd.DataFrame(chi2_matrix, columns=df.columns, index=df.columns)



def chi_square_heatmap(df):
    """
    Creates a heatmap based on the Chi-square test to evaluate the relationship between categorical variables.

    Arguments:
        df (pd.DataFrame): DataFrame with the categorical variables.

    Returns:
        None: Displays an interactive heatmap with Plotly).
    """
    chi_df = chi_square_matrix(df)

    # Create heatmap with Plotly
    fig = px.imshow(chi_df, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(color="Chi-cuadrado"),
                    title="Heatmap of Chi-square correlations between categorical variables")

    fig.update_layout(width=800, height=800)
    fig.show()