# DDoS Attack Analysis Report

## Log File
The log file is available https://github.com/elenesharia/aimlFin2026_e_sharia25/blob/main/task_3/e_sharia25_37142_server.log . It was downloaded from http://max.ge/aiml_final/e_sharia25_37142_server.log.

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
    pattern = r'(\S+) - - \[([^\]]+)\] "([^"]+)" (\d{3}) (\S+) "([^"]*)" "([^"]*)" (\d+)'
    match = re.match(pattern, line)
    if match:
        ip, timestamp_str, request, status, bytes_sent, referer, user_agent, extra = match.groups()
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S%z')
            return timestamp, ip, request, int(status)
        except ValueError:
            return None
    return None

```
---

### Regression and Anomaly Detection
```python
from scipy import stats

# After creating request_counts DataFrame...
request_counts['time_numeric'] = (request_counts['timestamp'] - request_counts['timestamp'].min()).dt.total_seconds() / 60

slope, intercept, r_value, p_value, std_err = stats.linregress(request_counts['time_numeric'], request_counts['count'])
request_counts['predicted'] = intercept + slope * request_counts['time_numeric']
request_counts['residual'] = request_counts['count'] - request_counts['predicted']

mean_res = request_counts['residual'].mean()
std_res = request_counts['residual'].std()
threshold = mean_res + 3 * std_res
anomalies = request_counts[request_counts['residual'] > threshold].copy()

anomalies['diff'] = anomalies['timestamp'].diff()
anomalies['group'] = (anomalies['diff'] > pd.Timedelta(minutes=1)).cumsum()
intervals = []
for group, data in anomalies.groupby('group'):
    start = data['timestamp'].min()
    end = data['timestamp'].max()
    intervals.append((start, end))

```
---

# Visualization Code
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(request_counts['timestamp'], request_counts['count'], label='Request Count')
plt.plot(request_counts['timestamp'], request_counts['predicted'], label='Regression Line', color='red')
for start, end in intervals:
    plt.axvspan(start, end, color='yellow', alpha=0.5, label='DDoS Interval' if intervals.index((start, end)) == 0 else "")
plt.xlabel('Time')
plt.ylabel('Requests per Minute')
plt.title('Request Volume with Regression and DDoS Anomalies')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ddos_visualization.png')
```
---
# Visualizations

The plot shows request counts over time, the linear regression trendline, and highlighted DDoS intervals in yellow.

<img width="1200" height="600" alt="ddos_visualization" src="https://github.com/user-attachments/assets/7cd0016d-4523-474b-95ba-63a379b34efc" />

The complete source code is available in the file - https://github.com/elenesharia/aimlFin2026_e_sharia25/blob/main/task_3/analyze_ddos.py

## Reproduction Steps

1. Clone/download the repository or navigate to the `task_3` folder.
2. Ensure Python 3.x and required libraries are installed:pip install pandas numpy scipy matplotlib
3. 3. Place the log file `e_sharia25_37142_server.log` in the same folder as the script.
4. Run the analysis script: python analyze_ddos.py
5. 5. The script will:
- Parse the log
- Perform linear regression
- Detect and print DDoS intervals
- Generate `ddos_visualization.png`


