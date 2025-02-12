import logging
import pandas as pd
import datapane as dp
from datetime import datetime

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Monitoring:
    def __init__(self):
        self.metrics = []

    def log_metric(self, metric_name, value):
        """Logs a metric for monitoring."""
        self.metrics.append(
            {"Metric": metric_name, "Value": value, "Timestamp": datetime.now()}
        )
        logger.info(f"Metric logged: {metric_name} = {value}")

    def generate_report(self, file_name="monitoring_report.html"):
        """Generates an interactive report using Datapane."""
        df = pd.DataFrame(self.metrics)

        report = dp.Report(
            dp.Table(df, caption="Pipeline Metrics Overview"),
            dp.Plot(
                df.plot(x="Timestamp", y="Value", kind="line", title="Metric Trends")
            ),
        )

        report.save(file_name)
        logger.info(f"Monitoring report saved as {file_name}")
