"""
CLI entry point: uv run python -m transcript_intelligence
or: transcript-intelligence (via pyproject.toml script)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv(".env", override=True)
console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]Transcript Intelligence[/bold cyan]\n"
        "Pipeline",
        border_style="cyan",
    ))

    # Validate environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not set. "
                      "Copy .env.example to .env and add your key.")
        sys.exit(1)

    data_dir = Path(os.getenv("DATA_DIR", "dataset"))
    if not data_dir.exists():
        console.print(f"[bold red]Error:[/bold red] DATA_DIR '{data_dir}' does not exist.")
        sys.exit(1)

    outputs_dir = Path(os.getenv("OUTPUTS_DIR", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "charts").mkdir(exist_ok=True)

    from transcript_intelligence.models import PipelineState
    from transcript_intelligence.pipeline.graph import build_graph

    graph = build_graph()
    initial_state: PipelineState = {}  # type: ignore[assignment]

    t0 = time.time()
    console.print("\n[bold]Starting pipeline...[/bold]\n")
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - t0

    console.print(f"\n[bold green]Pipeline completed in {elapsed:.1f}s[/bold green]")
    summary = final_state.get("summary", {})
    console.print(f"  Slide deck:  {final_state.get('report_path', 'N/A')}")
    console.print(f"  Results:     {final_state.get('results_json_path', 'N/A')}")
    console.print(f"  Charts:      {len(final_state.get('chart_paths', []))} files")


if __name__ == "__main__":
    main()
