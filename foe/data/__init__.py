"""
foe.data: BigQuery/GA4 connectivity and extraction -- foe's one deliberate
I/O exception (see the package README). Gated behind the `foe[bigquery]`
extra; importing foe.data itself never requires google-cloud-bigquery --
only instantiating DataEngine does.
"""

from foe.data.engine import DataEngine

__all__ = ["DataEngine"]
