FROM python:3.12-slim
WORKDIR /app

# Copy configuration, src and tests
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY tests/ ./tests/

# Install packages in editable mode inclduing dependencies
RUN pip install --no-cache-dir .[dev]

# Execute python
CMD ["python"]