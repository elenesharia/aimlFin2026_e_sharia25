# DDoS Attack Analysis Report

## Log File
The log file is available [here](e_sharia25_37142_server.log) (uploaded in this folder). It was downloaded from http://max.ge/aiml_final/e_sharia25_37142_server.log.

## Analysis Overview
- Parsed log entries to extract timestamps, IPs, requests, and status codes.
- Grouped requests into 1-minute intervals to compute request counts over time.
- Performed linear regression on the time series (minutes since start vs. request count) to model expected "normal" traffic.
- Calculated residuals (actual count - predicted count).
- Detected anomalies where residuals > mean residual + 3 * standard deviation of residuals (threshold ≈ 8414.68).
- Grouped contiguous anomaly minutes into attack intervals.
- Key stats: 61 minutes analyzed, mean requests/min ≈1507, max 13720 (indicating spikes). Regression slope ≈19.08 (slight upward trend), but low R² (0.014) suggests variability; used for baseline anomaly detection.

## Detected DDoS Time Intervals
- 2024-03-22 18:36:00+04:00 to 2024-03-22 18:37:00+04:00
- 2024-03-22 18:39:00+04:00 to 2024-03-22 18:40:00+04:00

These intervals show traffic spikes indicative of a DDoS attack.

## Main Code Fragments

### Log Parsing Function
```python
import re
from datetime import datetime

def parse_log_line(line):
    pattern = r'(\S+) - - $$   (.*?)   $$ "(.*?)" (\d+) (\S+) "(.*?)" "(.*?)" (\d+)'
    match = re.match(pattern, line)
    if match:
        ip, timestamp_str, request, status, bytes_sent, referer, user_agent, extra = match.groups()
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S%z')
            return timestamp, ip, request, int(status)
        except ValueError:
            return None
    return None
