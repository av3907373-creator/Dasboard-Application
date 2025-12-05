import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title='Amazon Product Analysis Dashboard', layout='wide')

# Load the preprocessed data
@st.cache_data
def load_data():
    df = pd.read_csv('amazon.csv')
    return df

df = load_data()

# Title of the dashboard
st.title('Amazon Product Analysis Dashboard')

# Sidebar for filters
st.sidebar.header('Filter Products')

# Rating slider
min_rating = df['rating'].min()
max_rating = df['rating'].max()
rating_selection = st.sidebar.slider(
    'Select a Rating Range',
    min_value=float(min_rating),
    max_value=float(max_rating),
    value=(float(min_rating), float(max_rating)),
    step=0.1
)

# Discount Percentage slider
min_discount = df['discount_percentage'].min()
max_discount = df['discount_percentage'].max()
discount_selection = st.sidebar.slider(
    'Select a Discount Percentage Range',
    min_value=float(min_discount),
    max_value=float(max_discount),
    value=(float(min_discount), float(max_discount)),
    step=1.0
)

# Discounted Price slider
min_price = df['discounted_price'].min()
max_price = df['discounted_price'].max()
price_selection = st.sidebar.slider(
    'Select a Discounted Price Range',
    min_value=float(min_price),
    max_value=float(max_price),
    value=(float(min_price), float(max_price)),
    step=10.0
)

# Main Category multiselect
available_categories = df['main_category'].unique().tolist()
selected_categories = st.sidebar.multiselect(
    'Select Main Categories',
    options=available_categories,
    default=available_categories
)

# Apply filters
filtered_df = df[
    (df['rating'] >= rating_selection[0]) &
    (df['rating'] <= rating_selection[1]) &
    (df['discount_percentage'] >= discount_selection[0]) &
    (df['discount_percentage'] <= discount_selection[1]) &
    (df['discounted_price'] >= price_selection[0]) &
    (df['discounted_price'] <= price_selection[1]) &
    (df['main_category'].isin(selected_categories))
]

# Display filtered data information
st.write(f"Displaying {len(filtered_df)} products out of {len(df)} total products.")

# Check if filtered_df is empty
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    # Product Price Analysis
    st.subheader('Product Price Analysis')
    fig_scatter = px.scatter(
        filtered_df,
        x='actual_price',
        y='discounted_price',
        color='main_category',
        hover_data=['product_name', 'rating', 'discount_percentage'],
        title='Actual Price vs. Discounted Price'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Product Category Distribution
    st.subheader('Product Category Distribution')

    # Count Plot
    st.write('#### Count of Products per Main Category')
    fig_count, ax_count = plt.subplots(figsize=(10, 6))
    sns.countplot(y='main_category', data=filtered_df, order=filtered_df['main_category'].value_counts().index, palette='viridis', ax=ax_count)
    ax_count.set_title('Distribution of Products Across Main Categories')
    ax_count.set_xlabel('Number of Products')
    ax_count.set_ylabel('Main Category')
    plt.tight_layout()
    st.pyplot(fig_count)

    # Pie Chart
    st.write('#### Percentage of Products per Main Category')
    category_counts = filtered_df['main_category'].value_counts().reset_index()
    category_counts.columns = ['main_category', 'count']
    fig_pie = px.pie(
        category_counts,
        values='count',
        names='main_category',
        title='Distribution of Main Categories',
        hole=0.3
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Generate Summary Report Button
    st.subheader('Summary Report')
    if st.button('Generate Summary Report'):
        st.write('### Filtered Data Summary Statistics')
        st.dataframe(filtered_df.describe())
        st.write('### First 10 Rows of Filtered Data')
        st.dataframe(filtered_df.head(10))

# Save the Streamlit app code to a file
streamlit_app_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title='Amazon Product Analysis Dashboard', layout='wide')

# Load the preprocessed data
@st.cache_data
def load_data():
    df = pd.read_csv('amazon.csv')
    return df

df = load_data()

# Title of the dashboard
st.title('Amazon Product Analysis Dashboard')

# Sidebar for filters
st.sidebar.header('Filter Products')

# Rating slider
min_rating = df['rating'].min()
max_rating = df['rating'].max()
rating_selection = st.sidebar.slider(
    'Select a Rating Range',
    min_value=float(min_rating),
    max_value=float(max_rating),
    value=(float(min_rating), float(max_rating)),
    step=0.1
)

# Discount Percentage slider
min_discount = df['discount_percentage'].min()
max_discount = df['discount_percentage'].max()
discount_selection = st.sidebar.slider(
    'Select a Discount Percentage Range',
    min_value=float(min_discount),
    max_value=float(max_discount),
    value=(float(min_discount), float(max_discount)),
    step=1.0
)

# Discounted Price slider
min_price = df['discounted_price'].min()
max_price = df['discounted_price'].max()
price_selection = st.sidebar.slider(
    'Select a Discounted Price Range',
    min_value=float(min_price),
    max_value=float(max_price),
    value=(float(min_price), float(max_price)),
    step=10.0
)

# Main Category multiselect
available_categories = df['main_category'].unique().tolist()
selected_categories = st.sidebar.multiselect(
    'Select Main Categories',
    options=available_categories,
    default=available_categories
)

# Apply filters
filtered_df = df[
    (df['rating'] >= rating_selection[0]) &
    (df['rating'] <= rating_selection[1]) &
    (df['discount_percentage'] >= discount_selection[0]) &
    (df['discount_percentage'] <= discount_selection[1]) &
    (df['discounted_price'] >= price_selection[0]) &
    (df['discounted_price'] <= price_selection[1]) &
    (df['main_category'].isin(selected_categories))
]

# Display filtered data information
st.write(f"Displaying {len(filtered_df)} products out of {len(df)} total products.")

# Check if filtered_df is empty
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    # Product Price Analysis
    st.subheader('Product Price Analysis')
    fig_scatter = px.scatter(
        filtered_df,
        x='actual_price',
        y='discounted_price',
        color='main_category',
        hover_data=['product_name', 'rating', 'discount_percentage'],
        title='Actual Price vs. Discounted Price'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Product Category Distribution
    st.subheader('Product Category Distribution')

    # Count Plot
    st.write('#### Count of Products per Main Category')
    fig_count, ax_count = plt.subplots(figsize=(10, 6))
    sns.countplot(y='main_category', data=filtered_df, order=filtered_df['main_category'].value_counts().index, palette='viridis', ax=ax_count)
    ax_count.set_title('Distribution of Products Across Main Categories')
    ax_count.set_xlabel('Number of Products')
    ax_count.set_ylabel('Main Category')
    plt.tight_layout()
    st.pyplot(fig_count)

    # Pie Chart
    st.write('#### Percentage of Products per Main Category')
    category_counts = filtered_df['main_category'].value_counts().reset_index()
    category_counts.columns = ['main_category', 'count']
    fig_pie = px.pie(
        category_counts,
        values='count',
        names='main_category',
        title='Distribution of Main Categories',
        hole=0.3
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Generate Summary Report Button
    st.subheader('Summary Report')
    if st.button('Generate Summary Report'):
        st.write('### Filtered Data Summary Statistics')
        st.dataframe(filtered_df.describe())
        st.write('### First 10 Rows of Filtered Data')
        st.dataframe(filtered_df.head(10))
'''

with open('streamlit_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("Streamlit app script 'streamlit_app.py' created.")
print("\nTo run the Streamlit app, execute the following command in your terminal:")
print("streamlit run streamlit_app.py")
