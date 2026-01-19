#!/bin/bash
# Installation script for reolink_AI

echo "🚀 Installing reolink_AI Watchdog System..."

# Check Python version
python3 --version | grep -q "3.11" || {
    echo "❌ Python 3.11 required"
    exit 1
}

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup database
echo "Setting up database..."
mysql -h 192.168.178.218 -u gh -p < ../config/schema.sql

# Create directories
mkdir -p /var/www/aufnahmen
mkdir -p /var/www/html/reports

# Copy config
cp ../config/config.yaml.example ../config/config.yaml
echo "⚠️  Edit config/config.yaml with your settings!"

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit config/config.yaml"
echo "2. Run: python src/watchdog.py"
