# Quick Start

fedviz is a lightweight monitoring toolkit for federated and distributed
training workflows.

## Installation

```bash
pip install -e .
```

## Basic usage

```python
import fedviz
from fedviz.emitters import SSEEmitter

fedviz.init(
    algorithm="FedAvg",
    emitters=[SSEEmitter(port=7070, serve_map=True)],
)
```

## Source layout

Recent refactors group related code into clearer subpackages:

- `src/fedviz/map/` for map metadata and map server code
- `src/fedviz/geo/` for IP parsing and geolocation helpers
- `src/fedviz/integrations/` for APPFL communicator patch helpers
