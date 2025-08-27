import sys
import json
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table

from .orchestrator import Orchestrator, OrchestratorConfig

console = Console()


@click.group()
@click.version_option()
def main():
    """PMS Orchestrator CLI."""


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), required=True, help="YAML config file")
def run(config_path: str):
    """Run an orchestrated PMS pipeline from YAML config."""
    try:
        cfg = OrchestratorConfig.from_yaml(Path(config_path))
        orch = Orchestrator(cfg)
        result = orch.run()
        table = Table(title="PMS Run Summary")
        table.add_column("Key")
        table.add_column("Value")
        for k, v in result.items():
            table.add_row(k, json.dumps(v, ensure_ascii=False)[:120])
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", highlight=False)
        raise SystemExit(1)


@main.command("show-config")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), required=True)
def show_config(config_path: str):
    """Pretty-print parsed config."""
    cfg = OrchestratorConfig.from_yaml(Path(config_path))
    console.print(cfg.model_dump(), soft_wrap=True)

if __name__ == "__main__":
    main()

