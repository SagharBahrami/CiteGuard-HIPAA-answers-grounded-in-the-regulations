FROM python:3.10-slim

WORKDIR /app

# The whole source tree is copied before install (rather than a
# dependencies-only layer first) because pip needs the actual module files
# present to build this package from src/.
#
# `pip install .` installs the citedguard package into site-packages, which is
# what the worker service resolves imports from: `rq worker` is a console
# script, so its sys.path[0] is the bin directory and /app is NOT importable
# from it. app.py and mcp_server.py stay unpackaged at /app and are launched
# by path, which does put /app on sys.path.
COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
