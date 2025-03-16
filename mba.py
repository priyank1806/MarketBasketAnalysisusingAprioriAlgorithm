import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit App Title
st.title("🛒 Market Basket Analysis using Apriori Algorithm")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your Basket CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Convert categorical data to dummy variables
    df1 = pd.get_dummies(df)
    df2 = df1.iloc[:, 1:]

    # Sidebar - User Input for Apriori Parameters
    st.sidebar.header("Apriori Parameters")
    min_support = st.sidebar.slider("Min Support", 0.01, 1.0, 0.2)
    min_confidence = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.2)
    min_lift = st.sidebar.slider("Min Lift", 1.0, 5.0, 1.0)

    # Apply Apriori Algorithm
    frequent_items = apriori(df2, min_support=min_support, use_colnames=True)

    # Fix: Ensure at least one frequent itemset exists
    if frequent_items.empty:
        st.warning("❌ No frequent itemsets found. Try lowering min support.")
    else:
        frequent_items['itemsets'] = frequent_items['itemsets'].apply(lambda x: ', '.join(list(x)))
        st.write("### Frequent Itemsets", frequent_items)

        # Generate Association Rules (🔥 Fix: Add support_only=True)
        try:
            rules = association_rules(frequent_items, metric="lift", min_threshold=min_lift, support_only=False)

            # Fix: Convert frozenset columns to strings
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

            # Filter Rules Based on User Input
            filtered_rules = rules[
                (rules['lift'] >= min_lift) &
                (rules['confidence'] >= min_confidence) &
                (rules['support'] >= min_support)
            ]

            # Display Rules
            st.write("### Filtered Association Rules", filtered_rules)

            # Scatter Plot: Support vs Confidence
            st.write("### Support vs Confidence Plot")
            plt.figure(figsize=(8, 6))
            plt.scatter(rules['support'], rules['confidence'], alpha=0.6)
            plt.xlabel('Support')
            plt.ylabel('Confidence')
            plt.title('Support vs Confidence')
            st.pyplot(plt)

            # Download Filtered Rules as CSV
            csv = filtered_rules.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Rules CSV", data=csv, file_name="association_rules.csv")

        except KeyError:
            st.error("❌ No association rules generated. Try adjusting your parameters.")
