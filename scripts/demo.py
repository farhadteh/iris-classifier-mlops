#!/usr/bin/env python3
"""
Demo script to showcase the MLflow Iris Classification Pipeline
"""

import requests
import json
import time
import subprocess
import os
from pathlib import Path

def print_banner(text):
    """Print a banner with text"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step, description):
    """Print a step with description"""
    print(f"\n🔸 Step {step}: {description}")

def check_service(url, service_name, timeout=30):
    """Check if a service is running"""
    print(f"Checking {service_name} at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is running!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    
    print(f"❌ {service_name} is not responding after {timeout}s")
    return False

def run_training_demo():
    """Run training demonstration"""
    print_banner("MLflow Iris Classification Pipeline Demo")
    
    print("🌸 Welcome to the MLflow Iris Classification Pipeline!")
    print("This demo will showcase the complete MLOps pipeline.")
    
    # Check if we're in the right directory
    if not Path("train.py").exists():
        print("❌ Error: Please run this demo from the MLFLOW directory")
        return False
    
    print_step(1, "Training Models")
    print("Training multiple models with hyperparameter tuning...")
    print("This will take a few minutes...")
    
    try:
        # Run training with reduced parameter space for demo
        result = subprocess.run([
            "python", "train.py", 
            "--models", "logistic_regression", "random_forest"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("📊 Models trained: Logistic Regression, Random Forest")
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training is taking longer than expected, but continuing demo...")
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    
    return True

def test_api_endpoints():
    """Test API endpoints"""
    print_step(2, "Testing API Endpoints")
    
    api_base = "http://localhost:8000"
    
    # Check if API is running
    if not check_service(f"{api_base}/health", "FastAPI"):
        print("Please start the FastAPI server first: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000")
        return False
    
    print("\n📡 Testing API endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{api_base}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test model info
    try:
        response = requests.get(f"{api_base}/model/info")
        if response.status_code == 200:
            print(f"✅ Model info: {response.status_code}")
            info = response.json()
            print(f"   Model: {info.get('model_name', 'Unknown')}")
            print(f"   Version: {info.get('model_version', 'Unknown')}")
        else:
            print(f"⚠️  Model info: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    # Test single prediction
    test_flower = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    try:
        response = requests.post(f"{api_base}/predict", json=test_flower)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single prediction: {response.status_code}")
            print(f"   Input: Sepal(5.1x3.5), Petal(1.4x0.2)")
            print(f"   Prediction: {result['class_name']} (confidence: {result['confidence']:.3f})")
        else:
            print(f"❌ Single prediction failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
    
    # Test batch prediction
    batch_data = {
        "instances": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.0, "sepal_width": 2.8, "petal_length": 4.0, "petal_width": 1.3},
            {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0}
        ]
    }
    
    try:
        response = requests.post(f"{api_base}/predict/batch", json=batch_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction: {response.status_code}")
            print(f"   Batch size: {result['batch_size']}")
            for i, pred in enumerate(result['predictions']):
                print(f"   Sample {i+1}: {pred['class_name']} (conf: {pred['confidence']:.3f})")
        else:
            print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
    
    return True

def check_services():
    """Check all services"""
    print_step(3, "Checking All Services")
    
    services = [
        ("MLflow UI", "http://localhost:5000"),
        ("FastAPI", "http://localhost:8000/health"),
        ("Streamlit", "http://localhost:8501/_stcore/health")
    ]
    
    running_services = []
    
    for name, url in services:
        if check_service(url, name, timeout=10):
            running_services.append(name)
    
    print(f"\n📊 Services Status: {len(running_services)}/{len(services)} running")
    
    if running_services:
        print("\n🌐 Access URLs:")
        if "MLflow UI" in running_services:
            print("   📈 MLflow UI: http://localhost:5000")
        if "FastAPI" in running_services:
            print("   🚀 FastAPI: http://localhost:8000")
            print("   📚 API Docs: http://localhost:8000/docs")
        if "Streamlit" in running_services:
            print("   🎨 Streamlit: http://localhost:8501")
    
    return len(running_services) > 0

def create_sample_data():
    """Create sample data for testing"""
    print_step(4, "Creating Sample Data")
    
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    sample_data = []
    
    # Generate samples for each class
    classes = ["setosa", "versicolor", "virginica"]
    for i, class_name in enumerate(classes):
        for _ in range(5):
            if class_name == "setosa":
                sample = {
                    "sepal_length": np.random.normal(5.0, 0.4),
                    "sepal_width": np.random.normal(3.4, 0.4),
                    "petal_length": np.random.normal(1.5, 0.2),
                    "petal_width": np.random.normal(0.2, 0.1),
                    "true_class": class_name
                }
            elif class_name == "versicolor":
                sample = {
                    "sepal_length": np.random.normal(6.0, 0.5),
                    "sepal_width": np.random.normal(2.8, 0.3),
                    "petal_length": np.random.normal(4.3, 0.5),
                    "petal_width": np.random.normal(1.3, 0.2),
                    "true_class": class_name
                }
            else:  # virginica
                sample = {
                    "sepal_length": np.random.normal(6.5, 0.6),
                    "sepal_width": np.random.normal(3.0, 0.3),
                    "petal_length": np.random.normal(5.5, 0.6),
                    "petal_width": np.random.normal(2.0, 0.3),
                    "true_class": class_name
                }
            
            # Ensure positive values
            for key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
                sample[key] = max(0.1, sample[key])
            
            sample_data.append(sample)
    
    # Save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_iris_data.csv", index=False)
    
    print("✅ Created sample_iris_data.csv with 15 samples")
    print("   You can upload this file to the Streamlit app for batch prediction")
    
    return True

def show_next_steps():
    """Show next steps for the user"""
    print_banner("Next Steps")
    
    print("🎉 Demo completed! Here's what you can do next:")
    print()
    print("1. 🔬 Explore MLflow UI:")
    print("   • View experiments and runs")
    print("   • Compare model performance")
    print("   • Manage model registry")
    print("   • URL: http://localhost:5000")
    print()
    print("2. 🚀 Test the API:")
    print("   • Interactive docs: http://localhost:8000/docs")
    print("   • Try different flower measurements")
    print("   • Test batch predictions")
    print()
    print("3. 🎨 Use the Streamlit App:")
    print("   • Interactive model testing")
    print("   • Upload CSV for batch prediction")
    print("   • Explore model behavior")
    print("   • URL: http://localhost:8501")
    print()
    print("4. 🐳 Try Docker deployment:")
    print("   docker-compose up -d")
    print()
    print("5. 🔄 Set up CI/CD:")
    print("   • Push to GitHub")
    print("   • Configure GitHub Actions")
    print("   • Enable automated training")
    print()
    print("📁 Files created:")
    print("   • sample_iris_data.csv - Sample data for testing")
    print("   • model_comparison.png - Model comparison plot")
    print("   • Various MLflow artifacts in mlruns/")

def main():
    """Main demo function"""
    print("🌸 Starting MLflow Iris Classification Demo...")
    
    try:
        # Step 1: Run training
        if run_training_demo():
            print("✅ Training demo completed")
        else:
            print("⚠️  Training demo had issues, but continuing...")
        
        time.sleep(2)
        
        # Step 2: Test API (if running)
        test_api_endpoints()
        
        time.sleep(1)
        
        # Step 3: Check services
        check_services()
        
        time.sleep(1)
        
        # Step 4: Create sample data
        create_sample_data()
        
        time.sleep(1)
        
        # Show next steps
        show_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
    
    print("\n🎯 Demo completed! Enjoy exploring your MLOps pipeline!")

if __name__ == "__main__":
    main()
