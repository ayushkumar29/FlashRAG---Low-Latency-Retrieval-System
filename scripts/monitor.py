import time
import requests
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from datetime import datetime


console = Console()


def get_metrics():
    try:
        response = requests.get("http://localhost:8000/api/metrics", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def create_metrics_table(metrics):
    table = Table(
        title="FlashRAG System Metrics",
        show_header=True,
        header_style="bold magenta",
        border_style="blue"
    )
    
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=20)
    table.add_column("Status", style="yellow", width=15)
    
    if metrics:
        table.add_row("Total Requests", str(metrics['total_requests']), "OK")
        table.add_row("Cache Hits", str(metrics['cache_hits']), "OK")
        table.add_row("Cache Misses", str(metrics['cache_misses']), "OK")
        
        cache_rate = float(metrics['cache_hit_rate'].rstrip('%'))
        cache_status = "Excellent" if cache_rate > 70 else "Low" if cache_rate > 40 else "Poor"
        table.add_row("Cache Hit Rate", metrics['cache_hit_rate'], cache_status)
        
        avg_lat = metrics['avg_latency']
        lat_status = "Fast" if avg_lat < 100 else "OK" if avg_lat < 500 else "Slow"
        table.add_row("Avg Latency", f"{avg_lat:.2f}ms", lat_status)
        
        rps = metrics['requests_per_second']
        rps_status = "High" if rps > 50 else "Medium" if rps > 10 else "Low"
        table.add_row("Requests/sec", f"{rps:.2f}", rps_status)
        
        uptime = time.time() - metrics['start_time']
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        table.add_row("Uptime", f"{hours:02d}:{minutes:02d}:{seconds:02d}", "OK")
    else:
        table.add_row("Status", "[red]Server Not Responding[/red]", "Error")
    
    return table


def create_dashboard(metrics):
    layout = Layout()
    
    header = Panel(
        f"[bold blue]FlashRAG Monitoring Dashboard[/bold blue]\n"
        f"[yellow]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/yellow]",
        border_style="green"
    )
    
    metrics_panel = Panel(
        create_metrics_table(metrics),
        border_style="blue"
    )
    
    footer = Panel(
        "[yellow]Press Ctrl+C to stop monitoring[/yellow] | "
        "[cyan]Refreshing every 1 second[/cyan]",
        border_style="green"
    )
    
    layout.split_column(
        Layout(header, size=3),
        Layout(metrics_panel),
        Layout(footer, size=3)
    )
    
    return layout


def monitor():
    console.clear()
    console.print("\n[bold blue]Starting FlashRAG Monitor...[/bold blue]\n")
    console.print("[yellow]Connecting to http://localhost:8000[/yellow]")
    console.print("[yellow]Make sure the server is running: python main.py serve[/yellow]\n")
    
    time.sleep(2)
    
    try:
        with Live(create_dashboard(None), refresh_per_second=1, console=console) as live:
            while True:
                metrics = get_metrics()
                live.update(create_dashboard(metrics))
                time.sleep(1)
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Monitoring stopped[/yellow]\n")


if __name__ == "__main__":
    monitor()
