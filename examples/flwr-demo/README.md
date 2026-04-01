# demo: A Flower / PYTORCH app

## Install dependencies and project

```bash
pip install -e .
```

## Run with the Simulation Engine

Start the mlflow server first:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

In the `flwr-demo` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

## Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
