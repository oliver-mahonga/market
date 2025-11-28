import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Configuration ---
st.set_page_config(
    page_title="Hybrid E-Commerce Recommender (Group 5)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. DATA LOADING, CLEANING, AND FEATURE ENGINEERING ---

@st.cache_data
def load_and_preprocess_data(file_name):
    """Loads, cleans, and engineers features from the retail data."""
    st.subheader("1. Data Processing: Loading and Cleaning...")
    
    try:
        # Load Excel file (correct for Online Retail dataset)
        if file_name.lower().endswith(".xlsx"):
            df = pd.read_excel(file_name)
        else:
            df = pd.read_csv(file_name, encoding='ISO-8859-1')
            
    except FileNotFoundError:
        st.error(f"Error: File '{file_name}' not found. Please ensure the data file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # Cleaning Steps
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[~df['InvoiceNo'].astype(str).str.contains('C', na=False)]
    df.drop_duplicates(inplace=True)
    df['Description'] = df['Description'].str.strip()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    # Time-of-Day Feature
    def get_time_of_day(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 21: return 'Evening'
        else: return 'Night'

    df['Time_of_Day'] = df['InvoiceDate'].dt.hour.apply(get_time_of_day)
    df['Day_of_Week'] = df['InvoiceDate'].dt.day_name()

    # CLV and Segmentation
    clv_df = df.groupby('CustomerID')['Revenue'].sum().reset_index().rename(columns={'Revenue':'CLV'})
    clv_df['Customer_Segment'] = pd.qcut(clv_df['CLV'], q=[0,0.33,0.66,1], labels=['Low_Value','Medium_Value','High_Value'])
    df = pd.merge(df, clv_df[['CustomerID','Customer_Segment']], on='CustomerID', how='left')

    st.success(f"Data Loaded and Cleaned. Total records: {len(df)}.")
    return df


# --- 2. MBA ENGINE: ASSOCIATION RULE MINING ---

@st.cache_data
def build_mba_engine(df):
    """Builds the Market Basket Analysis (MBA) model and generates rules."""
    st.subheader("2. MBA Engine: Generating Association Rules...")
    
    # Focus on UK data for performance and relevance
    df_mba = df[df['Country'] == 'United Kingdom'].copy()

    # Create transaction structure (InvoiceNo, Description)
    basket_raw = df_mba.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)

    # Encode units: 1 if quantity >= 1, 0 otherwise
    basket = (basket_raw >= 1)


    # Remove non-product codes (e.g., POSTAGE, D, M)
    non_products = [col for col in basket.columns if re.search(r'\b(POSTAGE|D|M|CR|CASH)\b', col, re.IGNORECASE) or col.strip() == '']
    basket.drop(columns=non_products, inplace=True, errors='ignore')

    # General MBA Rules (min_support=0.015)
    frequent_itemsets_gen = apriori(basket, min_support=0.015, use_colnames=True)
    rules_general = association_rules(frequent_itemsets_gen, metric="lift", min_threshold=1.0)
    
    # Filter General Rules (Advanced Metrics)
    rules_general_filtered = rules_general[
        (rules_general['lift'] >= 1.2) &
        (rules_general['conviction'] >= 1.1)
    ].sort_values('confidence', ascending=False).reset_index(drop=True)
    
    # High-Value Customer MBA Rules (min_support=0.015)
    high_value_invoices = df_mba[df_mba['Customer_Segment'] == 'High_Value']['InvoiceNo'].unique()
    basket_hv = basket[basket.index.isin(high_value_invoices)]
    
    frequent_itemsets_hv = apriori(basket_hv, min_support=0.015, use_colnames=True)
    rules_hv = association_rules(frequent_itemsets_hv, metric="lift", min_threshold=1.0)
    
    # Filter High-Value Rules
    rules_hv_filtered = rules_hv[
        (rules_hv['lift'] >= 1.2) &
        (rules_hv['conviction'] >= 1.1)
    ].sort_values('confidence', ascending=False).reset_index(drop=True)

    st.success(f"MBA Engine Built. General Rules: {len(rules_general_filtered)}, High-Value Rules: {len(rules_hv_filtered)}")
    return rules_general_filtered, rules_hv_filtered, basket.columns.tolist()

# --- 3. CBF ENGINE: CONTENT-BASED FILTERING ---

@st.cache_data
def build_cbf_engine(df):
    """Builds the Content-Based Filtering (CBF) model using TF-IDF and Cosine Similarity."""
    st.subheader("3. CBF Engine: Building Item Similarity Matrix...")

    # Get unique items and their descriptions
    item_features = df[['Description']].drop_duplicates().reset_index(drop=True)

    # Clean and filter descriptions
    item_features = item_features[
        ~item_features['Description'].str.contains('manual|postage|discount|set of|sale', case=False)
    ]
    item_features.dropna(inplace=True)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(item_features['Description'])
    
    # Cosine Similarity Calculation
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Description Index (mapping row/column to item name)
    desc_index = item_features['Description'].tolist()
    
    st.success(f"CBF Engine Built. Item feature matrix size: {tfidf_matrix.shape}")
    return cosine_sim, desc_index

# --- 4. HYBRID RECOMMENDATION LOGIC ---

def recommend_hybrid(item_description, customer_segment, num_recommendations, rules_general, rules_hv, cosine_sim, desc_index):
    """
    Blends recommendations from MBA (Rule-based) and CBF (Similarity-based).
    """
    final_recs = set()
    
    # 1. MBA Engine (Rule-Based)
    rules_set = rules_hv if customer_segment == 'High_Value' else rules_general

    mba_recs = []
    
    # Find all rules where the item is an antecedent
    for index, row in rules_set.iterrows():
        antecedents = str(row['antecedents']).replace("frozenset({'", '').replace("'})", '').split("', '")
        consequents = str(row['consequents']).replace("frozenset({'", '').replace("'})", '').split("', '")

        if item_description in antecedents:
            for consequent in consequents:
                if consequent not in final_recs and consequent != item_description:
                    mba_recs.append((consequent, row['lift']))
            
    # Sort MBA recs by Lift and take the top ones
    mba_recs_df = pd.DataFrame(mba_recs, columns=['item', 'lift']).drop_duplicates(subset=['item'])
    mba_recs_df = mba_recs_df.sort_values('lift', ascending=False)
    
    # Add top MBA recs to the final list (Priority 1)
    for item in mba_recs_df['item'].head(num_recommendations):
        final_recs.add(item)
        
    # 2. CBF Engine (Similarity-Based)
    cbf_recs = []
    
    if item_description in desc_index:
        idx = desc_index.index(item_description)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_similar_items = [i for i in sim_scores if desc_index[i[0]] != item_description]
        
        for i in top_similar_items:
            item = desc_index[i[0]]
            score = i[1]
            cbf_recs.append((item, score))
            
    cbf_recs_df = pd.DataFrame(cbf_recs, columns=['item', 'score']).drop_duplicates(subset=['item'])

    # 3. Blending and Final List
    final_list = list(final_recs)[:num_recommendations]

    # Fill remaining slots (Priority 2: CBF for novelty)
    for item in cbf_recs_df['item']:
        if item not in final_list and len(final_list) < num_recommendations:
            final_list.append(item)
        elif len(final_list) >= num_recommendations:
            break
    
    return final_list, mba_recs_df, cbf_recs_df

# --- 5. STREAMLIT APPLICATION UI ---

def main():
    
    # 5.1. Initialization (Run the Models)
    df = load_and_preprocess_data('Online Retail.xlsx')

    if df is None:
        st.stop()
        
    rules_general, rules_hv, unique_items = build_mba_engine(df)
    cosine_sim, desc_index = build_cbf_engine(df)

    # Get a list of items common to both MBA and CBF for the dropdown
    common_items = sorted(list(set(desc_index) & set(unique_items)))
    
    # 5.2. Title and Instructions
    st.title(" Hybrid Market Basket & Content Recommender")
    st.markdown("### Advanced ML Project for E-Commerce Personalization")
    st.markdown("---")

    # 5.3. Sidebar Controls
    st.sidebar.header("Recommendation Controls")
    selected_item = st.sidebar.selectbox(
        "1. Select an Item for Recommendation:",
        common_items
    )

    selected_segment = st.sidebar.radio(
        "2. Select Customer Segment:",
        ('High_Value', 'Medium_Value', 'Low_Value'),
        index=0,
        help="The MBA engine uses tailored rules for the 'High-Value' segment."
    )

    num_recs = st.sidebar.slider(
        "3. Number of Recommendations:",
        min_value=1, max_value=10, value=5
    )
    st.sidebar.markdown("---")

    # 5.4. Main App Execution
    if selected_item:
        st.header(f"Recommendations for: **{selected_item}**")
        
        # Run the Hybrid Model
        final_recommendations, mba_details, cbf_details = recommend_hybrid(
            selected_item, selected_segment, num_recs, 
            rules_general, rules_hv, cosine_sim, desc_index
        )
        
        # Display Final Recommendations
        st.subheader(f" Final Hybrid Recommendations ({len(final_recommendations)} Items)")
        
        cols = st.columns(num_recs)
        for i, rec_item in enumerate(final_recommendations):
            if i < num_recs:
                with cols[i]:
                    st.info(f"**Item {i+1}:**\n\n{rec_item}")
                
        st.markdown("---")
        
        # 5.5. Explainable AI (XAI) Analysis
        st.subheader(" Recommendation Engine Analysis (The WHY)")
        st.markdown("**This section shows how the two engines contribute to the final list.**")
        
        col1, col2 = st.columns(2)
        
        # Column 1: MBA Details
        with col1:
            st.markdown("#### Market Basket Analysis (MBA) Contribution")
            st.markdown(f"**Filter Used:** *Rules for **{selected_segment}** Segment*")
            
            if not mba_details.empty:
                st.dataframe(
                    mba_details.head(num_recs).rename(
                        columns={'item': 'Recommended Item', 'lift': 'Lift Score (Priority)'}
                    ).set_index('Recommended Item'),
                    use_container_width=True
                )
                st.caption("MBA prioritizes strong transactional associations (high Lift) for cross-selling.")
            else:
                st.warning("No high-confidence, high-lift MBA rules found for this item in this segment.")
                
        # Column 2: CBF Details
        with col2:
            st.markdown("#### Content-Based Filtering (CBF) Contribution")
            
            if not cbf_details.empty:
                st.dataframe(
                    cbf_details.head(num_recs).rename(
                        columns={'item': 'Recommended Item', 'score': 'Similarity Score (Novelty)'}
                    ).set_index('Recommended Item'),
                    use_container_width=True
                )
                st.caption("CBF provides novelty by recommending items with similar descriptions (high Cosine Similarity).")
            else:
                st.warning("Item not found in the CBF model index.")
                
        st.markdown("---")
        st.subheader(" Project Highlights (Advanced Features)")
        st.markdown(
            """
            - **Hybrid Blending:** Combines the transactional power of **MBA** with the novelty of **CBF**.
            - **Segmented Rules:** The MBA engine dynamically selects rules based on the **Customer Segment** (CLV).
            - **Explainable AI (XAI):** The analysis section shows the scores (Lift/Similarity) from each engine, demonstrating the model's rationale.
            """
        )

if __name__ == '__main__':
    main()