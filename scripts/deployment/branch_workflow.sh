#!/bin/bash

# MLOps Branch Workflow Manager
# This script helps manage the full development lifecycle

set -e

echo "🌊 MLOps Branch Workflow Manager"
echo "================================"

# Function to show current status
show_status() {
    echo ""
    echo "📊 Current Status:"
    echo "  • Current branch: $(git branch --show-current)"
    echo "  • Latest commit: $(git log --oneline -1)"
    echo "  • Remote branches:"
    git branch -r | grep -E "(main|develop)" || echo "    No remote branches found"
    echo ""
}

# Function to trigger staging deployment
trigger_staging() {
    echo "🧪 Triggering Staging Deployment"
    echo "================================"
    
    if [[ $(git branch --show-current) != "develop" ]]; then
        echo "⚠️  Warning: You're not on the develop branch!"
        echo "   Current branch: $(git branch --show-current)"
        echo "   Staging deploys from: develop"
        echo ""
        read -p "Do you want to switch to develop branch? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git checkout develop
            git pull origin develop
        else
            echo "❌ Staying on current branch. Staging won't trigger."
            return 1
        fi
    fi
    
    echo "🚀 Creating staging deployment commit..."
    echo "# Staging Deployment - $(date '+%Y-%m-%d %H:%M:%S')" >> STAGING_DEPLOYMENT.md
    echo "" >> STAGING_DEPLOYMENT.md
    echo "This commit triggers the staging deployment pipeline." >> STAGING_DEPLOYMENT.md
    echo "- Environment: staging" >> STAGING_DEPLOYMENT.md
    echo "- Branch: develop" >> STAGING_DEPLOYMENT.md
    echo "- Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> STAGING_DEPLOYMENT.md
    echo "" >> STAGING_DEPLOYMENT.md
    
    git add STAGING_DEPLOYMENT.md
    git commit -m "deploy: Trigger staging deployment $(date '+%Y-%m-%d %H:%M:%S')

- Update staging deployment marker
- Branch: develop → staging environment  
- Pipeline: test → build → deploy-staging
- Expected services: MLflow, FastAPI, Streamlit (staging)"
    
    git push origin develop
    
    echo "✅ Staging deployment triggered!"
    echo "🌐 Check GitHub Actions: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/actions"
}

# Function to trigger production deployment  
trigger_production() {
    echo "🚀 Triggering Production Deployment"
    echo "==================================="
    
    # Ensure we're on develop first
    if [[ $(git branch --show-current) != "develop" ]]; then
        echo "📋 Switching to develop branch first..."
        git checkout develop
        git pull origin develop
    fi
    
    # Switch to main and merge
    echo "🔄 Merging develop → main for production deployment..."
    git checkout main
    git pull origin main
    git merge develop --no-ff -m "release: Merge develop to main for production deployment

- Deploy tested features from develop to production
- Includes all staging-validated changes  
- Pipeline: test → build → deploy-production → monitor
- Expected services: MLflow, FastAPI, Streamlit (production)"
    
    git push origin main
    
    echo "✅ Production deployment triggered!"
    echo "🌐 Check GitHub Actions: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/actions"
}

# Function to show the complete workflow
show_workflow() {
    echo "🔄 Complete MLOps Workflow"
    echo "=========================="
    echo ""
    echo "📋 Branch Strategy:"
    echo "  develop  → 🧪 Staging Environment"
    echo "  main     → 🚀 Production Environment"
    echo ""
    echo "🌊 Development Flow:"
    echo "  1. Work on feature branches"
    echo "  2. Merge to develop → triggers staging deployment"
    echo "  3. Test in staging environment"
    echo "  4. Merge to main → triggers production deployment"
    echo "  5. Monitor production deployment"
    echo ""
    echo "🎯 CI/CD Pipeline Stages:"
    echo ""
    echo "  develop branch push:"
    echo "    ✅ test → ✅ build → ✅ deploy-staging"
    echo ""
    echo "  main branch push:"  
    echo "    ✅ test → ✅ build → ✅ deploy-production → ✅ monitor"
    echo ""
    echo "📚 Available Commands:"
    echo "  ./scripts/deployment/branch_workflow.sh staging     - Trigger staging"
    echo "  ./scripts/deployment/branch_workflow.sh production  - Trigger production" 
    echo "  ./scripts/deployment/branch_workflow.sh status      - Show current status"
    echo "  ./scripts/deployment/branch_workflow.sh workflow    - Show this help"
}

# Main script logic
case "$1" in
    "staging")
        show_status
        trigger_staging
        ;;
    "production")
        show_status
        trigger_production
        ;;
    "status")
        show_status
        ;;
    "workflow"|"help"|"")
        show_workflow
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_workflow
        exit 1
        ;;
esac
