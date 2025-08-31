#!/bin/bash

# MLflow Iris Classification - Service Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
SERVICE="all"
ENVIRONMENT="development"
DOCKER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--docker)
            DOCKER=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -s, --service    Service to start (all|mlflow|fastapi|streamlit|train)"
            echo "  -e, --env        Environment (development|production|testing)"
            echo "  -d, --docker     Use Docker containers"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Start all services in development mode"
            echo "  $0 -s fastapi               # Start only FastAPI service"
            echo "  $0 -d                        # Start all services with Docker"
            echo "  $0 -s train -e production   # Run training in production mode"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Starting MLflow Iris Classification services..."
print_status "Service: $SERVICE"
print_status "Environment: $ENVIRONMENT"
print_status "Docker: $DOCKER"

# Create necessary directories
mkdir -p mlruns models logs data

# Set environment variables
export ENVIRONMENT=$ENVIRONMENT

# Docker mode
if [ "$DOCKER" = true ]; then
    print_status "Using Docker containers..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    case $SERVICE in
        "all")
            print_status "Starting all services with Docker Compose..."
            docker-compose up -d mlflow-server fastapi-app streamlit-app
            ;;
        "mlflow")
            print_status "Starting MLflow server..."
            docker-compose up -d mlflow-server
            ;;
        "fastapi")
            print_status "Starting FastAPI server..."
            docker-compose up -d fastapi-app
            ;;
        "streamlit")
            print_status "Starting Streamlit app..."
            docker-compose up -d streamlit-app
            ;;
        "train")
            print_status "Running training pipeline..."
            docker-compose run --rm training
            ;;
        *)
            print_error "Unknown service: $SERVICE"
            exit 1
            ;;
    esac
    
    print_success "Docker services started!"
    print_status "Check status with: docker-compose ps"
    print_status "View logs with: docker-compose logs -f [service_name]"
    
else
    # Native mode
    print_status "Starting services natively..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        print_warning "No virtual environment found. Creating one..."
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        # Activate virtual environment
        if [ -d "venv" ]; then
            source venv/bin/activate
        elif [ -d ".venv" ]; then
            source .venv/bin/activate
        fi
    fi
    
    case $SERVICE in
        "all")
            print_status "Starting all services..."
            
            # Start MLflow server in background
            print_status "Starting MLflow server on port 5000..."
            mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns > logs/mlflow.log 2>&1 &
            MLFLOW_PID=$!
            echo $MLFLOW_PID > logs/mlflow.pid
            
            # Wait a bit for MLflow to start
            sleep 5
            
            # Start FastAPI server in background
            print_status "Starting FastAPI server on port 8000..."
            export MLFLOW_TRACKING_URI="http://localhost:5000"
            uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 > logs/fastapi.log 2>&1 &
            FASTAPI_PID=$!
            echo $FASTAPI_PID > logs/fastapi.pid
            
            # Wait a bit for FastAPI to start
            sleep 3
            
            # Start Streamlit app in background
            print_status "Starting Streamlit app on port 8501..."
            export API_BASE_URL="http://localhost:8000"
            streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &
            STREAMLIT_PID=$!
            echo $STREAMLIT_PID > logs/streamlit.pid
            
            print_success "All services started!"
            ;;
        "mlflow")
            print_status "Starting MLflow server on port 5000..."
            mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns
            ;;
        "fastapi")
            print_status "Starting FastAPI server on port 8000..."
            export MLFLOW_TRACKING_URI="http://localhost:5000"
            uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
            ;;
        "streamlit")
            print_status "Starting Streamlit app on port 8501..."
            export API_BASE_URL="http://localhost:8000"
            streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
            ;;
        "train")
            print_status "Running training pipeline..."
            export MLFLOW_TRACKING_URI="http://localhost:5000"
            python train.py
            ;;
        *)
            print_error "Unknown service: $SERVICE"
            exit 1
            ;;
    esac
fi

# Print service URLs
print_success "Services are starting up..."
print_status "Service URLs:"
print_status "  MLflow UI:    http://localhost:5000"
print_status "  FastAPI:      http://localhost:8000"
print_status "  FastAPI Docs: http://localhost:8000/docs"
print_status "  Streamlit:    http://localhost:8501"

# Create stop script
cat > stop_services.sh << 'EOF'
#!/bin/bash

# Stop all services

print_status() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

# Check if Docker mode
if [ -f "docker-compose.yml" ] && docker-compose ps | grep -q "Up"; then
    print_status "Stopping Docker services..."
    docker-compose down
    print_success "Docker services stopped!"
else
    print_status "Stopping native services..."
    
    # Stop services using PID files
    for service in mlflow fastapi streamlit; do
        if [ -f "logs/${service}.pid" ]; then
            pid=$(cat logs/${service}.pid)
            if kill -0 $pid 2>/dev/null; then
                kill $pid
                print_status "Stopped $service (PID: $pid)"
            fi
            rm -f logs/${service}.pid
        fi
    done
    
    # Kill any remaining processes
    pkill -f "mlflow server" 2>/dev/null || true
    pkill -f "uvicorn fastapi_app" 2>/dev/null || true
    pkill -f "streamlit run" 2>/dev/null || true
    
    print_success "All services stopped!"
fi
EOF

chmod +x stop_services.sh

print_status "To stop all services, run: ./stop_services.sh"
