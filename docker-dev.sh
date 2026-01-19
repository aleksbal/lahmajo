#!/bin/bash

# Docker development helper script for Lahmajo

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi
}

# Start services
start_services() {
    print_status "Starting Lahmajo services..."
    docker-compose up -d
    print_status "Services started successfully!"
    print_status "Web UI: http://localhost:8000"
    print_status "Elasticsearch: http://localhost:9200"
}

# Stop services
stop_services() {
    print_status "Stopping Lahmajo services..."
    docker-compose down
    print_status "Services stopped successfully!"
}

# View logs
view_logs() {
    if [ -n "$1" ]; then
        docker-compose logs -f "$1"
    else
        docker-compose logs -f
    fi
}

# Rebuild services
rebuild_services() {
    print_status "Rebuilding Lahmajo services..."
    docker-compose build --no-cache
    print_status "Rebuild completed!"
}

# Reset everything (including data)
reset_all() {
    print_warning "This will remove all containers, volumes, and data."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing all containers and volumes..."
        docker-compose down -v
        docker system prune -f
        print_status "Reset completed!"
    else
        print_status "Reset cancelled."
    fi
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Check Elasticsearch
    if curl -s http://localhost:9200/_cluster/health > /dev/null; then
        print_status "✓ Elasticsearch is healthy"
    else
        print_error "✗ Elasticsearch is not responding"
    fi
    
    # Check Lahmajo app
    if curl -s http://localhost:8000/documents > /dev/null; then
        print_status "✓ Lahmajo app is healthy"
    else
        print_error "✗ Lahmajo app is not responding"
    fi
}

# Show help
show_help() {
    echo "Lahmajo Docker Development Helper"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start      Start all services"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo "  logs       View logs (optional: specify service name)"
    echo "  rebuild    Rebuild all services"
    echo "  health     Check service health"
    echo "  reset      Reset everything (including data)"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start all services"
    echo "  $0 logs lahmajo             # View lahmajo app logs"
    echo "  $0 logs elasticsearch       # View elasticsearch logs"
    echo "  $0 health                   # Check service health"
}

# Main script logic
main() {
    check_docker
    check_docker_compose
    
    case "${1:-help}" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            start_services
            ;;
        logs)
            view_logs "$2"
            ;;
        rebuild)
            rebuild_services
            ;;
        health)
            check_health
            ;;
        reset)
            reset_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"