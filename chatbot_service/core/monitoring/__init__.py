"""
RAG System Monitoring Module

Provides comprehensive monitoring and observability for the RAG chatbot service.

Components:
  - prometheus_metrics.py: Metrics collection and Prometheus text export
  - grafana_dashboard.json: Pre-configured Grafana dashboard
  - prometheus_rules.yaml: Alert rules for anomaly detection

Quick Start:
  1. Import metrics:
     from core.monitoring.prometheus_metrics import get_metrics
     
  2. Record events:
     metrics = get_metrics()
     metrics.record_vector_search(duration_ms=150, num_results=5)
     metrics.record_compression(ratio=0.45, success=True)
  
  3. Export for Prometheus (FastAPI endpoint):
     @app.get("/metrics")
     def metrics():
         return Response(get_metrics().export_prometheus(), media_type="text/plain")
  
  4. Set up Prometheus scraping in prometheus.yml:
     global:
       scrape_interval: 15s
     scrape_configs:
       - job_name: 'rag-service'
         static_configs:
           - targets: ['localhost:8000']
  
  5. Import Grafana dashboard:
     - Open Grafana UI → Dashboards → Import
     - Upload grafana_dashboard.json
     - Select Prometheus data source
  
  6. Set up alert rules:
     - Copy prometheus_rules.yaml to Prometheus config directory
     - Add to prometheus.yml: rule_files: ["prometheus_rules.yaml"]
     - Restart Prometheus
"""

from .prometheus_metrics import (
    PrometheusMetrics,
    MetricType,
    MetricValue,
    get_metrics,
    reset_metrics,
)

__all__ = [
    "PrometheusMetrics",
    "MetricType",
    "MetricValue",
    "get_metrics",
    "reset_metrics",
]
