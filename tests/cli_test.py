from __future__ import annotations

from hivewatch import cli


class FakeMapServer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        FakeMapServer.instances.append(self)

    def start(self):
        self.started = True

    def serve_forever(self):
        return None

    def stop(self):
        self.stopped = True


def test_map_run_cli_watches_live_directory_when_no_run_id(monkeypatch):
    FakeMapServer.instances = []
    monkeypatch.setattr(cli, "MapServer", FakeMapServer)

    assert cli.main(["map", "run", "--runs-dir", "runs"]) == 0

    server = FakeMapServer.instances[0]
    assert server.started is True
    assert server.kwargs["watch"] is True
    assert server.kwargs["run_id"] is None


def test_map_run_cli_disables_watcher_for_static_run(monkeypatch):
    FakeMapServer.instances = []
    monkeypatch.setattr(cli, "MapServer", FakeMapServer)

    assert cli.main(["map", "run", "--runs-dir", "runs", "--run-id", "run-abc"]) == 0

    server = FakeMapServer.instances[0]
    assert server.kwargs["watch"] is False
    assert server.kwargs["run_id"] == "run-abc"
