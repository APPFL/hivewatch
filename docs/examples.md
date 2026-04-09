# Examples

## APPFL + map monitoring

The APPFL example under `examples/appfl/` now uses package-provided helpers
instead of a local `map_utils.py` file.

Relevant package modules:

- `fedviz.map` for map metadata generation and dashboard serving
- `fedviz.geo` for IP parsing and location resolution
- `fedviz.integrations` for communicator patching helpers used by the example

Run the example from `examples/appfl/`:

```bash
python run_server.py
python run_client.py --config ./resources/configs/client_1.yaml
python run_client.py --config ./resources/configs/client_2.yaml
```
