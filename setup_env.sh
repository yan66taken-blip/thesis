#!/bin/bash
# Creates .venv and installs all project dependencies.
# Run once from the project root: bash setup_env.sh

set -e

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

pip install \
  openai \
  langchain \
  langchain-core \
  langchain-openai \
  pydantic \
  pandas \
  numpy \
  matplotlib \
  seaborn \
  fastapi \
  "uvicorn[standard]"

echo ""
echo "Done. Activate with: source .venv/bin/activate"
