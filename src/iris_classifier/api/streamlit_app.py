import json
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Configure the page
st.set_page_config(
    page_title="Iris Classification App",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App configuration
API_BASE_URL = st.sidebar.text_input("API Base URL", value="http://localhost:8000")


def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, (
            response.json() if response.status_code == 200 else None
        )
    except Exception as e:
        return False, str(e)


def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        return response.status_code == 200, (
            response.json() if response.status_code == 200 else None
        )
    except Exception as e:
        return False, str(e)


def make_prediction(features):
    """Make a single prediction"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=features, timeout=10)
        return response.status_code == 200, (
            response.json() if response.status_code == 200 else response.text
        )
    except Exception as e:
        return False, str(e)


def make_batch_prediction(instances):
    """Make batch predictions"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch", json={"instances": instances}, timeout=30
        )
        return response.status_code == 200, (
            response.json() if response.status_code == 200 else response.text
        )
    except Exception as e:
        return False, str(e)


# Main app layout
st.title("🌸 Iris Classification App")
st.markdown("*Powered by MLflow and FastAPI*")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Single Prediction", "Batch Prediction", "Model Explorer", "API Status"],
)

# API Health Check
with st.sidebar.expander("🔧 API Health", expanded=False):
    if st.button("Check API Health"):
        healthy, health_data = check_api_health()
        if healthy:
            st.success("✅ API is healthy!")
            st.json(health_data)
        else:
            st.error(f"❌ API is not responding: {health_data}")

# Model Information
with st.sidebar.expander("📊 Model Info", expanded=False):
    if st.button("Get Model Info"):
        success, model_data = get_model_info()
        if success:
            st.success("✅ Model loaded!")
            st.json(model_data)
        else:
            st.error(f"❌ Could not get model info: {model_data}")

# Page content
if page == "Single Prediction":
    st.header("🔮 Single Prediction")
    st.markdown("Enter the iris flower measurements to get a prediction:")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sepal Measurements")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0, 0.1)

    with col2:
        st.subheader("Petal Measurements")
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.0, 0.1)

    # Feature visualization
    st.subheader("📏 Feature Values")
    feature_data = {
        "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Value": [sepal_length, sepal_width, petal_length, petal_width],
    }
    fig_features = px.bar(
        feature_data,
        x="Feature",
        y="Value",
        title="Input Feature Values",
        color="Value",
        color_continuous_scale="viridis",
    )
    st.plotly_chart(fig_features, use_container_width=True)

    # Make prediction
    if st.button("🔮 Predict", type="primary"):
        features = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }

        with st.spinner("Making prediction..."):
            success, result = make_prediction(features)

        if success:
            st.success("✅ Prediction successful!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Class", result["class_name"])
            with col2:
                st.metric("Class Index", result["prediction"])
            with col3:
                st.metric("Confidence", f"{result['confidence']:.3f}")

            # Probability visualization
            st.subheader("🎯 Prediction Probabilities")
            prob_data = pd.DataFrame(
                list(result["probabilities"].items()), columns=["Class", "Probability"]
            )

            fig_prob = px.bar(
                prob_data,
                x="Class",
                y="Probability",
                title="Class Probabilities",
                color="Probability",
                color_continuous_scale="RdYlGn",
            )
            fig_prob.update_layout(showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

            # Show raw result
            with st.expander("🔍 Raw Prediction Result"):
                st.json(result)
        else:
            st.error(f"❌ Prediction failed: {result}")

elif page == "Batch Prediction":
    st.header("📊 Batch Prediction")
    st.markdown(
        "Upload a CSV file or enter multiple measurements for batch prediction:"
    )

    # File upload option
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Data")
            st.dataframe(df)

            # Validate columns
            required_cols = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]
            if all(col in df.columns for col in required_cols):
                if st.button("🔮 Predict Batch", type="primary"):
                    instances = df[required_cols].to_dict("records")

                    with st.spinner(f"Making {len(instances)} predictions..."):
                        success, result = make_batch_prediction(instances)

                    if success:
                        st.success(
                            f"✅ Batch prediction successful! {result['batch_size']} predictions made."
                        )

                        # Convert results to DataFrame
                        predictions_df = pd.DataFrame(
                            [
                                {
                                    "prediction": pred["prediction"],
                                    "class_name": pred["class_name"],
                                    "confidence": pred["confidence"],
                                }
                                for pred in result["predictions"]
                            ]
                        )

                        # Combine with original data
                        result_df = pd.concat(
                            [df[required_cols], predictions_df], axis=1
                        )

                        st.subheader("📈 Prediction Results")
                        st.dataframe(result_df)

                        # Visualizations
                        col1, col2 = st.columns(2)

                        with col1:
                            # Class distribution
                            class_counts = predictions_df["class_name"].value_counts()
                            fig_dist = px.pie(
                                values=class_counts.values,
                                names=class_counts.index,
                                title="Predicted Class Distribution",
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                        with col2:
                            # Confidence distribution
                            fig_conf = px.histogram(
                                predictions_df,
                                x="confidence",
                                title="Confidence Distribution",
                                nbins=20,
                            )
                            st.plotly_chart(fig_conf, use_container_width=True)

                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"iris_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(f"❌ Batch prediction failed: {result}")
            else:
                st.error(f"❌ CSV must contain columns: {required_cols}")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")

    else:
        # Manual input option
        st.subheader("✏️ Manual Input")
        st.markdown("Enter measurements manually (add multiple rows):")

        if "batch_data" not in st.session_state:
            st.session_state.batch_data = []

        with st.form("batch_input_form"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                sepal_length = st.number_input(
                    "Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1
                )
            with col2:
                sepal_width = st.number_input(
                    "Sepal Width", min_value=0.0, max_value=10.0, value=3.0, step=0.1
                )
            with col3:
                petal_length = st.number_input(
                    "Petal Length", min_value=0.0, max_value=10.0, value=4.0, step=0.1
                )
            with col4:
                petal_width = st.number_input(
                    "Petal Width", min_value=0.0, max_value=10.0, value=1.0, step=0.1
                )

            submitted = st.form_submit_button("➕ Add to Batch")

            if submitted:
                new_row = {
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width,
                }
                st.session_state.batch_data.append(new_row)
                st.success("✅ Row added to batch!")

        if st.session_state.batch_data:
            st.subheader(f"📋 Batch Data ({len(st.session_state.batch_data)} rows)")
            batch_df = pd.DataFrame(st.session_state.batch_data)
            st.dataframe(batch_df)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔮 Predict Batch", type="primary"):
                    with st.spinner(
                        f"Making {len(st.session_state.batch_data)} predictions..."
                    ):
                        success, result = make_batch_prediction(
                            st.session_state.batch_data
                        )

                    if success:
                        st.success(f"✅ Batch prediction successful!")
                        st.json(result)
                    else:
                        st.error(f"❌ Batch prediction failed: {result}")

            with col2:
                if st.button("🗑️ Clear Batch"):
                    st.session_state.batch_data = []
                    st.success("✅ Batch cleared!")
                    st.experimental_rerun()

elif page == "Model Explorer":
    st.header("🔍 Model Explorer")
    st.markdown("Explore the iris dataset and model behavior:")

    # Generate sample data for exploration
    if st.button("🎲 Generate Sample Data"):
        import numpy as np

        np.random.seed(42)

        # Generate sample data points
        n_samples = 50
        sample_data = []

        # Generate samples for each class
        for class_idx, class_name in enumerate(["setosa", "versicolor", "virginica"]):
            for _ in range(n_samples // 3):
                if class_name == "setosa":
                    features = {
                        "sepal_length": np.random.normal(5.0, 0.4),
                        "sepal_width": np.random.normal(3.4, 0.4),
                        "petal_length": np.random.normal(1.5, 0.2),
                        "petal_width": np.random.normal(0.2, 0.1),
                    }
                elif class_name == "versicolor":
                    features = {
                        "sepal_length": np.random.normal(6.0, 0.5),
                        "sepal_width": np.random.normal(2.8, 0.3),
                        "petal_length": np.random.normal(4.3, 0.5),
                        "petal_width": np.random.normal(1.3, 0.2),
                    }
                else:  # virginica
                    features = {
                        "sepal_length": np.random.normal(6.5, 0.6),
                        "sepal_width": np.random.normal(3.0, 0.3),
                        "petal_length": np.random.normal(5.5, 0.6),
                        "petal_width": np.random.normal(2.0, 0.3),
                    }

                # Ensure positive values
                for key in features:
                    features[key] = max(0.1, features[key])

                sample_data.append(features)

        # Make batch predictions
        with st.spinner("Generating predictions for sample data..."):
            success, result = make_batch_prediction(sample_data)

        if success:
            # Create comprehensive DataFrame
            sample_df = pd.DataFrame(sample_data)
            predictions_df = pd.DataFrame(
                [
                    {
                        "predicted_class": pred["class_name"],
                        "confidence": pred["confidence"],
                        "prediction_idx": pred["prediction"],
                    }
                    for pred in result["predictions"]
                ]
            )

            full_df = pd.concat([sample_df, predictions_df], axis=1)

            st.subheader("📊 Sample Data with Predictions")
            st.dataframe(full_df)

            # Visualizations
            st.subheader("📈 Data Visualizations")

            # 3D scatter plot
            fig_3d = px.scatter_3d(
                full_df,
                x="sepal_length",
                y="sepal_width",
                z="petal_length",
                color="predicted_class",
                size="confidence",
                title="3D Scatter Plot: Sepal Length vs Width vs Petal Length",
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            # Feature correlation heatmap
            corr_matrix = sample_df.corr()
            fig_corr = px.imshow(
                corr_matrix, title="Feature Correlation Matrix", aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Feature distributions by class
            fig_dist = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Sepal Length",
                    "Sepal Width",
                    "Petal Length",
                    "Petal Width",
                ],
            )

            features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            colors = ["red", "green", "blue"]

            for i, feature in enumerate(features):
                row = i // 2 + 1
                col = i % 2 + 1

                for j, class_name in enumerate(["setosa", "versicolor", "virginica"]):
                    class_data = full_df[full_df["predicted_class"] == class_name][
                        feature
                    ]
                    fig_dist.add_trace(
                        go.Histogram(
                            x=class_data,
                            name=f"{class_name}",
                            marker_color=colors[j],
                            opacity=0.7,
                        ),
                        row=row,
                        col=col,
                    )

            fig_dist.update_layout(
                title="Feature Distributions by Predicted Class", height=600
            )
            st.plotly_chart(fig_dist, use_container_width=True)

elif page == "API Status":
    st.header("🔧 API Status & Management")

    # Real-time status monitoring
    status_placeholder = st.empty()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Status"):
            pass  # Status will be updated below

    with col2:
        if st.button("🔄 Reload Model"):
            with st.spinner("Reloading model..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/model/reload", timeout=30)
                    if response.status_code == 200:
                        st.success("✅ Model reloaded successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Model reload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error reloading model: {e}")

    with col3:
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)")

    # Status monitoring
    while True:
        with status_placeholder.container():
            st.subheader("📊 Current Status")

            # API Health
            healthy, health_data = check_api_health()
            if healthy:
                st.success("✅ API is healthy")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", health_data.get("status", "Unknown"))
                with col2:
                    st.metric(
                        "Model Status", health_data.get("model_status", "Unknown")
                    )

                # Model info
                success, model_data = get_model_info()
                if success:
                    st.subheader("🤖 Model Information")

                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.metric("Model Name", model_data.get("model_name", "Unknown"))
                        st.metric(
                            "Model Version", model_data.get("model_version", "Unknown")
                        )
                    with info_col2:
                        st.metric(
                            "Model Stage", model_data.get("model_stage", "Unknown")
                        )
                        st.metric(
                            "Run ID",
                            model_data.get("mlflow_run_id", "Unknown")[:8] + "...",
                        )

                    with st.expander("🔍 Full Model Info"):
                        st.json(model_data)
                else:
                    st.warning("⚠️ Could not get model information")

            else:
                st.error(f"❌ API is not responding: {health_data}")

        if not auto_refresh:
            break

        time.sleep(5)
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit, FastAPI, and MLflow*")
