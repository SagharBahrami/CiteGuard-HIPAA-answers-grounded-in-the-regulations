FROM python:3.10-slim

WORKDIR /app

# The whole source tree is copied before install (rather than a
# dependencies-only layer first) because pip needs the actual module files
# present to build this package -- pyproject.toml lists py-modules/packages
# explicitly rather than being auto-discovered from a requirements.txt.
COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
