import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    /* Main background handled by config.toml, but we can add subtle gradients if needed */
    
    h1, h2, h3 {
        color: #3498db !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric Card Styling for Dark Mode */
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border-left: 5px solid #3498db;
        border: 1px solid #464e5f;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #3498db;
    }
    .metric-label {
        font-size: 16px;
        color: #e0e0e0;
        margin-bottom: 5px;
    }
    
    /* Sidebar adjustments if needed */
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_resource
def load_model_and_pipeline():
    """Loads the trained model and preprocessing pipeline."""
    try:
        model = joblib.load('model.pkl')
        pipeline = joblib.load('pipeline.pkl')
        return model, pipeline
    except Exception as e:
        st.error(f"Error loading model or pipeline: {e}")
        return None, None

@st.cache_data
def load_data():
    """Loads the housing dataset for the dashboard."""
    try:
        df = pd.read_csv('housing.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def predict_price(model, pipeline, input_data):
    """Runs the prediction pipeline."""
    try:
        # Preprocess the input data
        # Assuming the pipeline expects a DataFrame
        processed_data = pipeline.transform(input_data)
        prediction = model.predict(processed_data)
        return prediction[0]
    except Exception as e:
        # Fallback if pipeline expects raw data or behaves differently
        try:
             prediction = model.predict(input_data)
             return prediction[0]
        except Exception as e2:
            st.error(f"Prediction Error: {e2}")
            return None

# --- Main App ---

def main():
    # Load resources
    model, pipeline = load_model_and_pipeline()
    df = load_data()

    # Sidebar
    st.sidebar.header("🏡 Property Details")
    st.sidebar.markdown("Enter the features of the house to estimate its price.")
    
    with st.sidebar.form("prediction_form"):
        st.subheader("Geographic Location")
        longitude = st.number_input("Longitude", value=-119.5, min_value=-125.0, max_value=-114.0, step=0.1)
        latitude = st.number_input("Latitude", value=35.6, min_value=32.0, max_value=42.0, step=0.1)
        
        st.subheader("Housing Details")
        housing_median_age = st.slider("Median Age (Years)", 1, 52, 29)
        total_rooms = st.number_input("Total Rooms", value=2600, min_value=1, step=10)
        total_bedrooms = st.number_input("Total Bedrooms", value=530, min_value=1, step=10)
        
        st.subheader("Demographics")
        population = st.number_input("Population", value=1400, min_value=1, step=10)
        households = st.number_input("Households", value=500, min_value=1, step=10)
        median_income = st.number_input("Median Income (Tens of $000s)", value=3.8, min_value=0.0, max_value=15.0, step=0.1, help="e.g., 3.8 means $38,000")
        
        st.subheader("Location Type")
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
        )
        
        submit_button = st.form_submit_button("Predict Price 🚀")

    # Main Content Area
    st.title("California Housing Price Predictor")
    st.markdown("### 🤖 AI-Powered Real Estate Valuation")
    
    # Tabs for Layout
    tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Data Insights Dashboard", "ℹ️ About the Project"])

    # --- Tab 1: Prediction ---
    with tab1:
        if submit_button:
            if model is not None and pipeline is not None:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'longitude': [longitude],
                    'latitude': [latitude],
                    'housing_median_age': [housing_median_age],
                    'total_rooms': [total_rooms],
                    'total_bedrooms': [total_bedrooms],
                    'population': [population],
                    'households': [households],
                    'median_income': [median_income],
                    'ocean_proximity': [ocean_proximity]
                })

                with st.spinner("Calculating estimated value..."):
                    prediction = predict_price(model, pipeline, input_data)

                if prediction is not None:
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Estimated Median House Value</div>
                                <div class="metric-value">${prediction:,.2f}</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    st.markdown("---")
                    
                    # Optional: Show input summary
                    with st.expander("See Input Summary"):
                        st.dataframe(input_data)
            else:
                st.error("Model or Pipeline not loaded. Please check the files.")
        else:
            st.info("👈 Adjust the property details in the sidebar and click 'Predict Price' to see the result.")
            
            # Placeholder content
            st.markdown("#### Why use this tool?")
            st.markdown("""
            - **Accurate Estimates**: Powered by an advanced XGBoost model.
            - **Instant Results**: Get real-time valuations based on market data.
            - **Easy to Use**: Simple interface for quick inputs.
            """)

    # --- Tab 2: Dashboard ---
    with tab2:
        if df is not None:
            st.header("Data Insights Dashboard")
            
            # Row 1
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.subheader("Distribution of House Values")
                fig_hist = px.histogram(
                    df, 
                    x="median_house_value", 
                    nbins=50, 
                    title="Histogram of Median House Values",
                    color_discrete_sequence=['#3498db']
                )
                fig_hist.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col_d2:
                st.subheader("Correlation Heatmap")
                # Select only numeric columns for correlation
                numeric_df = df.select_dtypes(include=[np.number])
                corr = numeric_df.corr()
                fig_corr = px.imshow(
                    corr, 
                    text_auto=True, 
                    aspect="auto", 
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # Row 2
            col_d3, col_d4 = st.columns(2)
            
            with col_d3:
                st.subheader("Income vs. House Value")
                # Sample data for scatter plot to improve performance if dataset is large
                sample_df = df.sample(n=min(2000, len(df)), random_state=42)
                fig_scatter = px.scatter(
                    sample_df, 
                    x="median_income", 
                    y="median_house_value", 
                    color="ocean_proximity",
                    title="Median Income vs House Value (Sampled)",
                    opacity=0.6
                )
                fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col_d4:
                st.subheader("Price by Ocean Proximity")
                avg_price = df.groupby("ocean_proximity")["median_house_value"].mean().reset_index()
                fig_bar = px.bar(
                    avg_price, 
                    x="ocean_proximity", 
                    y="median_house_value", 
                    color="ocean_proximity",
                    title="Average House Value per Location",
                    text_auto='.2s'
                )
                fig_bar.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Dataset 'housing.csv' not found. Dashboard cannot be displayed.")

    # --- Tab 3: About ---
    with tab3:
        st.header("About the Project")
        
        st.markdown("""
        ### 📊 Dataset Summary
        This project uses the **California Housing dataset**, which contains data drawn from the 1990 U.S. Census. It includes metrics such as population, median income, and median house price for each block group in California.
        
        ### 🛠 Feature Engineering
        To prepare the data for the model, the following steps were likely taken (based on standard pipelines):
        - **Handling Missing Values**: Imputing missing counts for `total_bedrooms`.
        - **Categorical Encoding**: One-hot encoding the `ocean_proximity` feature.
        - **Feature Scaling**: Standardizing numerical features to ensure the model treats them equally.
        - **New Features**: Potentially creating features like `rooms_per_household` or `bedrooms_per_room`.
        
        ### 🚀 Model Choice: XGBoost
        We utilized **XGBoost (Extreme Gradient Boosting)** for this regression task.
        - **Why it works**: XGBoost is highly efficient and accurate for structured tabular data. It handles non-linear relationships well and is robust to outliers.
        - **Performance**: It typically outperforms simpler models like Linear Regression or Decision Trees on this type of dataset.
        
        ### 🌍 Impact
        Accurate housing price predictions are crucial for:
        - **Real Estate Agents**: To price homes competitively.
        - **Home Buyers**: To assess fair market value.
        - **Urban Planners**: To understand housing affordability trends.
        """)
        
        st.info("Built with Streamlit & Python")

if __name__ == "__main__":
    main()
