#!/bin/bash
# ============================================================
# Graph-Augmented Table Retrieval in RAG Pipelines
# Project Structure Setup Script
# ============================================================

echo "🔧 Setting up project structure..."

# Root-level directories
mkdir -p src/{data,pipelines,graph,evaluation,utils}
mkdir -p src/pipelines/{text_baseline,image_baseline,graph_augmented,shared}
mkdir -p data/{raw,processed,embeddings,results}
mkdir -p data/raw/{wikitablequestions,tatqa,finqa}
mkdir -p data/processed/{wikitablequestions,tatqa,finqa}
mkdir -p data/embeddings/{text,image,graph}
mkdir -p data/results/{pipeline1_text,pipeline2_image,pipeline3_graph}
mkdir -p configs
mkdir -p notebooks
mkdir -p tests
mkdir -p docs/{figures,tables}
mkdir -p scripts
mkdir -p logs

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/data/__init__.py
touch src/pipelines/__init__.py
touch src/pipelines/text_baseline/__init__.py
touch src/pipelines/image_baseline/__init__.py
touch src/pipelines/graph_augmented/__init__.py
touch src/pipelines/shared/__init__.py
touch src/graph/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

echo "✅ Directory structure created."
echo ""
echo "📁 Project structure:"
find . -type f -o -type d | head -80 | sort
