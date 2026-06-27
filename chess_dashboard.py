import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, mean_squared_error, r2_score)
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="Chess Ratings Dashboard", layout="wide")
st.title("♟️ FIDE World Chess Ratings - ML Dashboard")

# Load and prepare data
@st.cache_data
def load_data():
    # Load the CSV file
    df = pd.read_csv('03_FIDE_Chess_Ratings.csv')
    
    # Rename columns to match what the code expects
    df = df.rename(columns={
        'fed': 'Federation',
        'rating': 'Rating',
        'games': 'Games',
        'bday': 'Birth_Year'
    })
    
    # Clean the data
    # Remove rows with missing values in important columns
    df = df.dropna(subset=['Rating', 'Federation'])
    
    # If Title column has missing values, fill with 'None'
    if 'title' in df.columns:
        df['Title'] = df['title'].fillna('None')
    else:
        df['Title'] = 'None'
    
    # Calculate age from birth year
    if 'Birth_Year' in df.columns:
        df['Age'] = 2026 - df['Birth_Year']
        # Remove unreasonable ages
        df = df[(df['Age'] >= 5) & (df['Age'] <= 100)]
    else:
        # Generate random age if not available
        np.random.seed(42)
        df['Age'] = np.random.randint(10, 70, len(df))
    
    # Clean Games column
    if 'Games' in df.columns:
        df['Games'] = df['Games'].fillna(0)
    
    # Remove rows with unrealistic ratings
    df = df[(df['Rating'] >= 800) & (df['Rating'] <= 2900)]
    
    # Encode categorical variables
    le_fed = LabelEncoder()
    le_title = LabelEncoder()
    
    df['Federation_Code'] = le_fed.fit_transform(df['Federation'].astype(str))
    df['Title_Code'] = le_title.fit_transform(df['Title'].astype(str))
    
    # Binary target: Is GM?
    df['Is_GM'] = (df['Title'] == 'GM').astype(int)
    
    return df, le_fed, le_title

# Load data
df, le_fed, le_title = load_data()

# Sidebar
st.sidebar.header("⚙️ Controls")
model_choice = st.sidebar.selectbox(
    "Select ML Model to View",
    [
        "📊 All Models Overview",
        "1️⃣ Rating Prediction (Regression)",
        "2️⃣ GM Classification (Binary)",
        "3️⃣ Title Classification (Multi-class)",
        "4️⃣ Federation Clustering"
    ]
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.3, 0.05)
use_scaling = st.sidebar.checkbox("Apply Scaling", True)

# Show dataset info
st.sidebar.markdown("---")
st.sidebar.write(f"📊 Dataset: {len(df)} players")
st.sidebar.write(f"🏷️ Titles: {', '.join(df['Title'].unique())}")

# --- Helper Functions ---

def prepare_data(target_col, feature_cols):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle stratification for classification
    stratify_param = y if len(y.unique()) < 10 and len(y.unique()) > 1 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_param
    )
    
    if use_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# --- 1. ALL MODELS OVERVIEW ---
if model_choice == "📊 All Models Overview":
    st.header("📊 Machine Learning Models Overview")
    
    st.markdown("""
    This dashboard shows **4 different ML applications** for chess data:
    
    1. **Rating Prediction** - Predict player's FIDE rating
    2. **GM Classification** - Predict if player will become Grandmaster
    3. **Title Classification** - Predict player's title (GM/IM/FM/CM/None)
    4. **Federation Clustering** - Group countries by playing strength
    """)
    
    # Show sample data
    st.subheader("📋 Data Sample")
    st.dataframe(df[['name', 'Federation', 'Rating', 'Age', 'Games', 'Title']].head(10))
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(df))
    with col2:
        st.metric("Avg Rating", f"{df['Rating'].mean():.0f}")
    with col3:
        st.metric("GM Count", df[df['Title'] == 'GM'].shape[0])
    with col4:
        st.metric("Avg Age", f"{df['Age'].mean():.1f}")
    
    # Add some visualizations
    st.subheader("📊 Rating Distribution by Title")
    fig = px.box(df, x='Title', y='Rating', color='Title', 
                 title='Rating Distribution by Chess Title')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Top 10 Federations by Average Rating")
    top_feds = df.groupby('Federation')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_feds, x='Federation', y='Rating', 
                 title='Top 10 Federations by Average Rating')
    st.plotly_chart(fig, use_container_width=True)

# --- 2. RATING PREDICTION (Regression) ---
elif model_choice == "1️⃣ Rating Prediction (Regression)":
    st.header("🎯 1. Rating Prediction (Regression)")
    
    st.markdown("""
    **Goal**: Predict a player's FIDE rating based on their profile.
    **Model**: Random Forest Regressor
    """)
    
    # Features
    features = ['Age', 'Games', 'Federation_Code']
    target = 'Rating'
    
    X_train, X_test, y_train, y_test = prepare_data(target, features)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
        st.info("Higher is better (max 1.0)")
    with col2:
        st.metric("RMSE", f"{rmse:.1f}")
        st.info("Lower is better")
    
    # Plot predictions
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Rating', 'y': 'Predicted Rating'},
        title=f"Actual vs Predicted Ratings (R² = {r2:.3f})"
    )
    fig.add_trace(px.line(x=[800, 2900], y=[800, 2900]).data[0])
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🔑 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': ['Age', 'Games', 'Federation'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# --- 3. GM CLASSIFICATION (Binary) ---
elif model_choice == "2️⃣ GM Classification (Binary)":
    st.header("🏆 2. GM Classification (Binary)")
    
    st.markdown("""
    **Goal**: Predict if a player will become a Grandmaster.
    **Model**: Logistic Regression
    """)
    
    # Features
    features = ['Rating', 'Age', 'Games', 'Federation_Code']
    target = 'Is_GM'
    
    X_train, X_test, y_train, y_test = prepare_data(target, features)
    
    # Balance check
    st.info(f"GM Percentage in dataset: {df['Is_GM'].mean()*100:.1f}%")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    st.metric("Accuracy", f"{accuracy:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['Not GM', 'Is GM'],
        y=['Not GM', 'Is GM'],
        colorscale='Blues'
    )
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=['Not GM', 'Is GM']))
    
    # Feature importance (coefficients)
    st.subheader("🔑 Feature Impact")
    coef_df = pd.DataFrame({
        'Feature': ['Rating', 'Age', 'Games', 'Federation'],
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=True)
    
    fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# --- 4. TITLE CLASSIFICATION (Multi-class) ---
elif model_choice == "3️⃣ Title Classification (Multi-class)":
    st.header("🎖️ 3. Title Classification (Multi-class)")
    
    st.markdown("""
    **Goal**: Predict player's exact title (GM, IM, FM, CM, or None).
    **Model**: Random Forest Classifier
    """)
    
    # Features
    features = ['Rating', 'Age', 'Games', 'Federation_Code']
    target = 'Title'
    
    X_train, X_test, y_train, y_test = prepare_data(target, features)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy:.3f}")
    
    # Confusion Matrix
    labels = sorted(df['Title'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues'
    )
    fig.update_layout(title="Confusion Matrix - Title Prediction")
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    # Feature importance
    st.subheader("🔑 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': ['Rating', 'Age', 'Games', 'Federation'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# --- 5. FEDERATION CLUSTERING ---
elif model_choice == "4️⃣ Federation Clustering":
    st.header("🌍 4. Federation Performance Analysis")
    
    st.markdown("""
    **Goal**: Group federations by player performance and development.
    **Method**: K-means Clustering on aggregated stats
    """)
    
    # Aggregate by federation
    fed_stats = df.groupby('Federation').agg({
        'Rating': 'mean',
        'Games': 'mean',
        'Age': 'mean',
        'Title_Code': 'mean',
        'name': 'count'  # Number of players
    }).rename(columns={'name': 'Player_Count'}).reset_index()
    
    # Scale features
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(fed_stats[['Rating', 'Games', 'Title_Code']])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    fed_stats['Cluster'] = kmeans.fit_predict(scaled_stats)
    
    # Visualization
    fig = px.scatter(
        fed_stats,
        x='Rating',
        y='Games',
        size='Player_Count',
        color=fed_stats['Cluster'].astype(str),
        text='Federation',
        title='Federation Clusters (Rating vs Games)',
        labels={'Cluster': 'Cluster'}
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster profiles
    st.subheader("📊 Cluster Profiles")
    
    cols = st.columns(3)
    cluster_names = ['Strong Elites', 'Developing', 'Emerging']
    
    for i in range(3):
        cluster_data = fed_stats[fed_stats['Cluster'] == i]
        with cols[i]:
            st.write(f"**{cluster_names[i]}**")
            st.write(f"Countries: {len(cluster_data)}")
            if len(cluster_data) > 0:
                st.write(f"Avg Rating: {cluster_data['Rating'].mean():.0f}")
                st.write(f"Avg Games: {cluster_data['Games'].mean():.0f}")
                st.write(f"Players: {cluster_data['Player_Count'].sum():,}")
    
    # Show data
    st.subheader("Federation Data")
    st.dataframe(fed_stats.sort_values('Rating', ascending=False))

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("💡 Select different ML models from the dropdown above to explore various predictions.")
