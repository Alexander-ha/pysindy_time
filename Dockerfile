FROM python:3.12-slim

WORKDIR /app/pysindy/

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install .

RUN python -c "import pysindy; print(f'PySINDy {pysindy.__version__} installed successfully')"

ENV EXAMPLE=example_fixed.py
CMD python examples/$EXAMPLE
