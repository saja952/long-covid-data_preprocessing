import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Long COVID Data Dashboard",
    page_icon="🦠",
    layout="wide"
)

def load_data():
    df = pd.read_csv("Post-COVID_Conditions.csv")
    return df


if "df" not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.original_df = load_data().copy()

df = st.session_state.df.copy()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [" Overview", " Missing Values", " Encoding", " Visualization", " Relationships", " Save / Reset"]
)

if page == " Overview":
    st.title("🦠 Long COVID Data Analysis Dashboard")

    st.markdown("""
    ### About this dataset:
    This dataset provides detailed estimates on **Post-COVID Conditions (Long COVID)** across the United States.  
    It includes information from the **U.S. Census Bureau's Household Pulse Survey**, reporting:
    - The percentage of adults who have **ever experienced Long COVID**  
    - Those **currently experiencing** it  
    - The **impact of symptoms** on daily life  

    The data is categorized by demographics such as **age, sex, gender identity, race, education**, and **state**.
    """)

    st.divider()
    st.header("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown(f"**Total Rows:** {df.shape[0]}  **Total Columns:** {df.shape[1]}")

    st.divider()
    st.header("Explore a Column")
    column = st.selectbox("Select a column to explore:", df.columns)
    if column:
        st.write("**Data type:**", df[column].dtype)
        st.write("**Missing values:**", df[column].isnull().sum())
        st.write("**Unique values:**", df[column].nunique())
        st.write("**Example values:**", df[column].unique()[:10])
        if df[column].dtype != 'object':
            st.write(df[column].describe())

elif page == " Missing Values":
    st.title(" Missing Values Handling")
    st.markdown("""
    Choose how to handle missing data in your dataset.  
    You can fill them using different statistical methods or a custom value.
    """)

    missing = df.isnull().sum()
    st.write("### Missing values per column:")
    st.write(missing[missing > 0])

    if missing.sum() > 0:
        fill_method = st.selectbox(
            "Select filling method:",
            ["None", "Mean", "Median", "Mode", "Custom value"]
        )

        if fill_method != "None":
            modified_df = df.copy()
            for col in modified_df.columns:
                if modified_df[col].isnull().sum() > 0:
                    if fill_method == "Mean" and modified_df[col].dtype != 'object':
                        modified_df[col] = modified_df[col].fillna(modified_df[col].mean())
                    elif fill_method == "Median" and modified_df[col].dtype != 'object':
                        modified_df[col] = modified_df[col].fillna(modified_df[col].median())
                    elif fill_method == "Mode":
                        modified_df[col] = modified_df[col].fillna(modified_df[col].mode()[0])
                    elif fill_method == "Custom value":
                        value = st.text_input(f"Enter custom value for {col}:")
                        if value:
                            modified_df[col] = modified_df[col].fillna(value)

            st.session_state.df = modified_df
            df = modified_df
            st.success(" Missing values handled successfully!")
            st.dataframe(df.head())

    else:
        st.info("No missing values detected!")

elif page == " Encoding":
    st.title(" Data Encoding")
    st.markdown("""
    Convert categorical (text-based) columns into numeric format using Label or One-Hot Encoding.
    """)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    selected_cols = st.multiselect("Select columns to encode:", categorical_cols)

    if selected_cols:
        encoding_type = st.radio("Choose encoding type:", ["Label Encoding", "One-Hot Encoding"])

        if st.button("Apply Encoding"):
            encoded_df = df.copy()
            if encoding_type == "Label Encoding":
                le = LabelEncoder()
                for col in selected_cols:
                    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            else:
                encoded_df = pd.get_dummies(encoded_df, columns=selected_cols)

            st.session_state.df = encoded_df
            df = encoded_df
            st.success(" Encoding applied successfully!")
            st.dataframe(df.head())

elif page == " Visualization":
    st.title(" Data Visualization")
    st.markdown("Visualize distributions and relationships between variables in your dataset.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    chart_type = st.selectbox("Choose chart type:", ["Histogram", "Boxplot", "Correlation Heatmap", "Scatter Plot"])

    if chart_type == "Histogram":
        col = st.selectbox("Select numeric column:", num_cols)
        if col:
            fig, ax = plt.subplots()
            sns.histplot(df[col], bins=20, kde=True, ax=ax)
            st.pyplot(fig)

    elif chart_type == "Boxplot":
        col = st.selectbox("Select numeric column:", num_cols)
        if col:
            fig, ax = plt.subplots()
            sns.boxplot(df[col], ax=ax)
            st.pyplot(fig)

    elif chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis:", num_cols)
        y_col = st.selectbox("Select Y-axis:", num_cols)
        if x_col and y_col:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
            st.pyplot(fig)

elif page == " Relationships":
    st.title(" Relationship Identification")
    st.markdown("""
    Analyze how variables relate to each other to better understand patterns in the Long COVID data.
    """)

    if "Value" in df.columns:
        group_col = st.selectbox("Select a grouping column:", df.columns)
        if group_col:
            rel_df = df.groupby(group_col)["Value"].mean().sort_values(ascending=False)
            st.bar_chart(rel_df)
            st.markdown(f" This shows how **Long COVID rates** vary by **{group_col}**.")
    else:
        st.warning("Column 'Value' not found in dataset!")

elif page == " Save / Reset":
    st.title(" Save or Reset Processed Data")

    st.subheader("Download the cleaned and encoded dataset")
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="Processed_LongCOVID_Data.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader(" Reset to Original Data")
    if st.button("Reset Dataset"):
        st.session_state.df = st.session_state.original_df.copy()
        st.success(" Dataset has been reset to its original state!")
