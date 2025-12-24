import time
import requests
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

def get_metrics():
    """Fetch metrics from API"""
    try:
        response = requests.get("http://localhost:8000/api/metrics", timeout=2)
        return response.json()
    except:
        return None

def create_table(metrics):
    """Create metrics table"""
    table = Table(title="FlashRAG Live Metrics", show_header=True, header_style="bold magenta")
    
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="green", width=20)
    
    if metrics:
        table.add_row("Total Requests", str(metrics['total_requests']))
        table.add_row("Cache Hits", str(metrics['cache_hits']))
        table.add_row("Cache Misses", str(metrics['cache_misses']))
        table.add_row("Cache Hit Rate", metrics['cache_hit_rate'])
        table.add_row("Avg Latency", f"{metrics['avg_latency']:.2f}ms")
        table.add_row("Requests/sec", f"{metrics['requests_per_second']:.2f}")
    else:
        table.add_row("Status", "[red]Server not responding[/red]")
    
    return table

def monitor():
    """Live monitoring dashboard"""
    console.print("\n[bold blue]FlashRAG Monitoring Dashboard[/bold blue]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")
    
    try:
        with Live(create_table(None), refresh_per_second=1, console=console) as live:
            while True:
                metrics = get_metrics()
                live.update(create_table(metrics))
                time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")

if __name__ == "__main__":
    monitor()
