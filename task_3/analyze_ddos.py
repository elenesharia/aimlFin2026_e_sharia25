import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Parse log line
def parse_log_line(line):
    pattern = r'(\S+) - - \[(.*?)\] "(.*?)" (\d+) (\S+) "(.*?)" "(.*?)" (\d+)'
    match = re.match(pattern, line)
    if match:
        ip, timestamp_str, request, status, bytes_sent, referer, user_agent, extra = match.groups()
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S%z')
            return timestamp, ip, request, int(status)
        except ValueError:
            return None
    return None

# Read and parse
logs = []
with open('e_sharia25_37142_server.log', 'r') as f:
    for line in f:
        parsed = parse_log_line(line.strip())
        if parsed:
            logs.append(parsed)

# DataFrame
df = pd.DataFrame(logs, columns=['timestamp', 'ip', 'request', 'status'])
df.set_index('timestamp', inplace=True)

# Resample to 1 min counts
request_counts = df.resample('1min').size().reset_index(name='count')  # Use '1min' instead of '1T'
request_counts['time_numeric'] = (request_counts['timestamp'] - request_counts['timestamp'].min()).dt.total_seconds() / 60

# Linear regression
if len(request_counts) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(request_counts['time_numeric'], request_counts['count'])
    request_counts['predicted'] = intercept + slope * request_counts['time_numeric']
    request_counts['residual'] = request_counts['count'] - request_counts['predicted']

    # Anomalies
    mean_res = request_counts['residual'].mean()
    std_res = request_counts['residual'].std()
    threshold = mean_res + 3 * std_res
    anomalies = request_counts[request_counts['residual'] > threshold].copy()

    # Groups
    if not anomalies.empty:
        anomalies.sort_values('timestamp', inplace=True)
        anomalies['diff'] = anomalies['timestamp'].diff()
        anomalies['group'] = (anomalies['diff'] > pd.Timedelta(minutes=1)).cumsum()
        intervals = []
        for group, data in anomalies.groupby('group'):
            start = data['timestamp'].min()
            end = data['timestamp'].max()
            intervals.append((start, end))
            print(f"DDoS interval: {start} to {end}")
    else:
        print("No anomalies detected.")
else:
    print("Insufficient data for regression.")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(request_counts['timestamp'], request_counts['count'], label='Request Count')
plt.plot(request_counts['timestamp'], request_counts['predicted'], label='Regression Line', color='red')
if 'intervals' in locals():
    for i, (start, end) in enumerate(intervals):
        plt.axvspan(start, end, color='yellow', alpha=0.5, label='DDoS Interval' if i == 0 else "")
plt.xlabel('Time')
plt.ylabel('Requests per Minute')
plt.title('Request Volume with Regression and DDoS Anomalies')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ddos_visualization.png')
plt.show()
