import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('CO2_emission.csv')

# Title of the app
st.title("CO2 Emissions Analysis")

# Display the head and tail of the DataFrame
st.subheader("Data Preview")
st.write("Head of the DataFrame:")
st.dataframe(df.head())
st.write("Tail of the DataFrame:")
st.dataframe(df.tail())

# Descriptive statistics for numerical fields
st.subheader("Descriptive Statistics")
numeric_columns = df.select_dtypes(include=[np.number]).columns
st.write(df[numeric_columns].describe())

# Data Clean-up
import streamlit as st
import pandas as pd

# Load data
st.subheader('CO2 Emissions Data Cleanup')

# Display initial data
st.write('### Initial Data')
st.dataframe(df.head())

# Check if the columns '2019' and '2019.1' are equal
if df['2019'].equals(df['2019.1']):
    st.write("Columns '2019' and '2019.1' are equal")
    df.drop(['2019.1'], axis='columns', inplace=True)

# Replace NaN values with 0 for the '2019' column
df['2019'] = df['2019'].fillna(0)

# Data Cleanup: Remove rows with NaN values
df_cleaned = df.dropna()

# Melt the DataFrame for easier analysis
df_melted = df_cleaned.melt(id_vars=['Country Name', 'country_code', 'Region'],
                            var_name='Year', value_name='CO2 Emissions')

# Convert Year to integer for plotting and analysis
df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
df_melted = df_melted.dropna(subset=['Year'])
df_melted['Year'] = df_melted['Year'].astype(int)

# Display intermediate data after cleanup
st.write('### Intermediate Data After Cleanup')
st.dataframe(df_melted.head())

# Check for missing values
st.write("\nMissing values after cleanup:")
st.write(df.isnull().sum())

# Drop rows with NaN values and reset index
df = df.dropna()
df = df.reset_index(drop=True)

st.write("\nMissing values after dropping and resetting index:")
st.write(df.isnull().sum())

# Check for duplicated countries
duplicated_country = df[df.duplicated(["Country Name"])]
st.write("Duplicated Countries:")
st.write(duplicated_country)

st.write("The data is now devoid of any null values and ready for analysis.")

st.subheader("Plots")

# Top 5 countries with highest CO2 emissions per capita in 2019
top_countries = df.nlargest(5, '2019')[['Country Name', '2019']]
st.subheader("Top 5 Countries with Highest CO2 Emissions per Capita in 2019")
st.dataframe(top_countries)

# Plots for numerical fields
st.subheader("Histograms of CO2 Emissions Per Capita")
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, year in enumerate(['1990', '2000', '2019']):
    axs[i].hist(df[year].dropna(), bins=30)
    axs[i].set_title(f'CO2 emissions per capita in {year}')
    axs[i].set_xlabel('CO2 emissions (metric tons per capita)')
    axs[i].set_ylabel('Frequency')

plt.tight_layout()
st.pyplot(fig)

st.subheader("Data visualisation")

# Data Analysis and Visualization
st.write('### Pairplot of CO2 Emissions per Capita Across Years')

sns.set_palette("husl")
fig = sns.pairplot(df[['1990', '2000', '2010', '2019']], diag_kind='kde')
plt.suptitle('Pairplot of CO2 Emissions per Capita Across Years', y=1.02)

# Display the plot in Streamlit
st.pyplot(fig)

# Description
st.write("""
The pairplot shows the distribution of CO2 emissions per capita for the years 1990, 2000, 2010, and 2019. 
Each diagonal plot represents the distribution for a specific year, allowing us to see how emissions are spread across different countries.
""")

st.write('### Correlation Heatmap')

# Create the correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df[['1990', '2000', '2010', '2019']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=.5, linecolor='black')
plt.title('Correlation Heatmap of CO2 Emissions per Capita Across Years')

# Display the plot in Streamlit
st.pyplot(plt)

# Description
st.write("""
The heatmap helps identify whether there are consistent trends over time. If correlations between consecutive years (e.g., 1990-2000, 2000-2010) are strong, it indicates that emission levels are relatively stable or growing at a consistent rate.
If there is a noticeable drop in correlation between certain years (for example, from 2010 to 2019), it may suggest changes in policies or practices affecting emissions.
""")

# Box Plot for CO2 emissions per capita by year
# Melt the DataFrame for easier analysis
df_melted = df.melt(id_vars=['Country Name', 'country_code', 'Region'],
                    var_name='Year', value_name='CO2_Emissions')  # Changed value_name to 'CO2_Emissions'

# Create a box plot for CO2 emissions per capita by year
fig = px.box(df_melted, x='Year', y='CO2_Emissions',
             color='Year',  # Color by year for distinction
             title='Box Plot of CO2 Emissions per Capita by Year (1990-2019)',
             labels={'CO2_Emissions': 'CO2 Emissions (metric tons per capita)', 'Year': 'Year'},
             color_discrete_sequence=px.colors.qualitative.Plotly)  # Use a qualitative color palette

# Display the plot in Streamlit
st.plotly_chart(fig)

# Description
st.write("""
### Distribution of Emissions
The box plot shows the median, quartiles, and potential outliers for CO2 emissions per capita across different years.

Trends Over Time: By comparing the boxes across years, you can observe how emissions have changed over time, including any shifts in median values or changes in variability.

Outliers: Any points outside the whiskers indicate countries with exceptionally high or low emissions relative to others in that year.
""")

# Bar Plot for Countries with Highest CO2 Emissions in 2019

top_15 = df.groupby('Country Name')['2019'].sum().sort_values(ascending=False)[0:15]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(top_15.index, top_15.values, color=sns.color_palette("viridis", 15))

ax.plot(top_15.index, top_15.values, marker='o', linestyle='-', linewidth=2, markersize=8, color='red')

ax.set_title('Countries with Highest CO2 Emissions (Metric Tons per Capita)', fontsize=16, fontweight='bold')
ax.set_ylabel('CO2 Emissions (Metric Tons per Capita) in 2019', fontsize=12)
ax.set_xticklabels(top_15.index, rotation=45, ha='right', fontsize=10)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# Observations
st.write("""
### Observations:
- Highest Average CO2 Emission Over Time: The plot reveals some interesting findings. While the USA maintains the highest total CO2 emissions over time, it ranks second in per capita emissions. China and Russia drop out of the top 20 countries in the per capita emissions plot.
- Top Emitters: A surprising observation is that Qatar claims the top spot, despite its population in 2019 being only 2.688 million.

Let's analyze the top country for per capita emissions over time.
""")

# Melt the DataFrame for easier analysis
df_melted = df.melt(id_vars=['Country Name', 'country_code', 'Region', 'Indicator Name'],
                    var_name='Year', value_name='CO2 Emissions')

# Convert Year to numeric and filter out rows with zero or NaN emissions
df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
df_melted['CO2 Emissions'] = pd.to_numeric(df_melted['CO2 Emissions'], errors='coerce')
df_filtered = df_melted[(df_melted['CO2 Emissions'] > 0) & (~df_melted['Country Name'].isna())]

# Find the country with the highest per capita emissions for each year
top_country_per_capita_year = df_filtered.loc[df_filtered.groupby('Year')['CO2 Emissions'].idxmax()]

# Select relevant columns for the final table
top_country_table = top_country_per_capita_year[['Year', 'Country Name', 'CO2 Emissions']]

# Reset index for better formatting
top_country_table.reset_index(drop=True, inplace=True)

# Display the resulting table
st.write("### Country with the Highest Per Capita Emissions for Each Year")
st.dataframe(top_country_table)

# Create a line plot based on the results
fig = px.line(top_country_table, x='Year', y='CO2 Emissions', color='Country Name',
              title='Top Country for CO2 Emissions per Capita Over Time',
              labels={'CO2 Emissions': 'CO2 Emissions (metric tons per capita)', 'Year': 'Year'},
              markers=True)

# Display the plot in Streamlit
st.plotly_chart(fig)

# Observations
st.write("""
### Observations:
1. Dominance of the UAE: The United Arab Emirates consistently ranks as the top country for CO2 emissions per capita across the years from 1990 to 2019.
2. High Emission Levels: The UAE's emissions per capita are significantly higher than those of other countries, often exceeding values of over 30 metric tons in the early years.
3. Trend Over Time: While the UAE has maintained its position as the highest emitter, there is a noticeable decline in emissions per capita from a peak of 31.78 metric tons in 1991 to 19.33 metric tons in 2019.
4. Implications for Policy: The decrease in emissions over time could suggest effective environmental policies or shifts towards more sustainable energy practices in the UAE, despite its high initial emissions.
""")
# Choropleth Map for CO2 Emission by Country Code and Year
fig_choropleth = px.choropleth(
    df_melted,
    locations="country_code",
    color="CO2 Emissions",
    hover_name="Country Name",
    hover_data=["CO2 Emissions"],
    title="CO2 Emission (metric tons per capita)",
    color_continuous_scale=px.colors.sequential.Plasma,
    animation_frame='Year',
)
fig_choropleth.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_choropleth)

st.subheader("Hypothesis & Checking")
st.write("The trend of CO2 emissions per capita shows significant regional variation, "
         "with developed regions exhibiting a stabilization or decline in emissions over the years, "
         "while developing regions continue to see an increase.")

# Data Cleanup: Remove rows with NaN values
df_cleaned = df.dropna()

# Melt the DataFrame to have years as a variable for easier analysis
df_melted = df_cleaned.melt(id_vars=['Country Name', 'country_code', 'Region', 'Indicator Name'],
                            var_name='Year', value_name='CO2 Emissions')

# Convert Year to integer for plotting and analysis
df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
df_melted = df_melted.dropna(subset=['Year'])
df_melted['Year'] = df_melted['Year'].astype(int)

# Calculate average CO2 emissions per region over the years
regional_trends = df_melted.groupby(['Year', 'Region'])['CO2 Emissions'].mean().reset_index()

# Plotting the trends of CO2 emissions per capita by region
st.write('### Average CO2 Emissions per Capita by Region (1990-2019)')
fig, ax = plt.subplots(figsize=(14, 8))
sns.lineplot(data=regional_trends, x='Year', y='CO2 Emissions', hue='Region', marker='o', ax=ax)
ax.set_title('Average CO2 Emissions per Capita by Region (1990-2019)')
ax.set_xlabel('Year')
ax.set_ylabel('Average CO2 Emissions (metric tons per capita)')
plt.xticks(rotation=45)
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, alpha=0.3)

# Display the plot in Streamlit
st.pyplot(fig)

# Observations
st.write("""
### Observations:
The plot illustrates that there is significant variation in CO2 emissions per capita across different countries and regions. Developed countries like the United States, Canada, and Australia tend to have much higher per capita emissions (often above 15 metric tons) compared to developing countries in regions like Sub-Saharan Africa and South Asia, which generally have emissions below 1 metric ton per capita. This disparity highlights the complex relationship between economic development and carbon emissions, suggesting that as countries industrialize and increase energy consumption, their CO2 emissions tend to rise. However, some developed nations have shown a slight decline in emissions in recent years, possibly due to increased environmental regulations and adoption of cleaner technologies.
""")

# Load data
st.subheader('CO2 Emissions Data Transformation')

# Melt the DataFrame for easier analysis
df_melted = df.melt(id_vars=['Country Name', 'country_code', 'Region', 'Indicator Name'],
                    var_name='Year', value_name='CO2 Emissions')

# Create a new column for Total CO2 Emissions by summing emissions across all years
total_emissions = df_melted.groupby('Country Name')['CO2 Emissions'].sum().reset_index()
total_emissions.rename(columns={'CO2 Emissions': 'Total CO2 Emissions'}, inplace=True)

# Merge the total emissions back into the original melted dataframe
df_melted = df_melted.merge(total_emissions, on='Country Name')

# Create a new column for Average CO2 Emissions by calculating mean across all years
df_melted['Average CO2 Emissions'] = df_melted.groupby('Country Name')['CO2 Emissions'].transform('mean')

# Display the modified DataFrame with new columns
st.write("### Modified DataFrame with Total and Average CO2 Emissions")
st.dataframe(df_melted[['Country Name', 'Year', 'CO2 Emissions', 'Total CO2 Emissions', 'Average CO2 Emissions']].head())

# Observations
st.write("""
### Observations:
1. **Total CO2 Emissions**: A new column, Total CO2 Emissions, is created by grouping the data by country and summing the emissions across all years. This provides insight into the total emissions produced by each country over the entire period.
2. **Merging Total Emissions**: The total emissions are merged back into the melted DataFrame, allowing each row to have access to its respective country's total emissions.
3. **Average CO2 Emissions Column**: Another new column, Average CO2 Emissions, is calculated by taking the mean of emissions for each country across all years. This helps in understanding how emissions per capita vary on average over time.
""")

st.subheader('Summary')
st.write("""
### 
Global CO2 emissions trends, as revealed by a comprehensive dataset, merit thoughtful consideration. Upon surveying data spanning 1990 through 2019, several discernible patterns emerge. Fossil fuels, predominantly coal and oil, undoubtedly factor prominently into emissions totals. Certain countries, including the United States and China, chronically appear atop rankings of highest-emitting locales. Illuminating as well, smaller Gulf states like the United Arab Emirates and Qatar paradoxically top charts of per capita outputs, underscoring the carbon-intensity of their economies reliant on abundant resources. All signify change occurs gradually, necessitating cross-national cooperation and sustained mitigation efforts proportionate to both present contributions and future capacity to reduce impacts on worldwide atmospheric concentrations.
""")