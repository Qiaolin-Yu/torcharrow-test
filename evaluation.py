import torcharrow as ta
from torcharrow import functional
import time
import numpy as np
import plotly.graph_objects as go

def test_sigrid_hash(batch_sizes, rounds=5):
    sigrid_times = []
    for batch_size in batch_sizes:
        times = []
        for _ in range(rounds):
            data = np.random.randint(1, 1000000, size=batch_size)
            col = ta.column(data)
            salt = 0  # As in the example
            max_value = 100  # As in the example
            start_time = time.time()
            hashed_col = functional.sigrid_hash(col, salt=salt, max_value=max_value)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        avg_time = sum(times) / rounds
        sigrid_times.append(avg_time)
        print(f"sigrid_hash - Batch size: {batch_size}, Average Time over {rounds} runs: {avg_time:.3f} milliseconds")
    fig = go.Figure(data=go.Scatter(x=batch_sizes, y=sigrid_times, mode='lines+markers'))
    fig.update_layout(title='Average Processing Time vs Batch Size for sigrid_hash',
                      xaxis_title='Batch Size',
                      yaxis_title='Average Processing Time (milliseconds)')
    fig.write_image('sigrid_hash_performance.png')

def test_bucketize(batch_sizes, rounds=5):
    bucketize_times = []
    for batch_size in batch_sizes:
        times = []
        for _ in range(rounds):
            # Generate data similar to the example: values between 1 and 11
            data = np.random.randint(1, 12, size=batch_size)
            col = ta.column(data)
            borders = [2, 5, 10]  # As in the example
            start_time = time.time()
            bucketized_col = functional.bucketize(col, borders)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        avg_time = sum(times) / rounds
        bucketize_times.append(avg_time)
        print(f"bucketize - Batch size: {batch_size}, Average Time over {rounds} runs: {avg_time:.3f} milliseconds")
    fig = go.Figure(data=go.Scatter(x=batch_sizes, y=bucketize_times, mode='lines+markers'))
    fig.update_layout(title='Average Processing Time vs Batch Size for bucketize',
                      xaxis_title='Batch Size',
                      yaxis_title='Average Processing Time (milliseconds)')
    fig.write_image('bucketize_performance.png')

def test_log(batch_sizes, rounds=5):
    log_times = []
    for batch_size in batch_sizes:
        times = []
        for _ in range(rounds):
            data = np.random.rand(batch_size) * 100 + 1  # Ensure data is positive
            col = ta.column(data)
            start_time = time.time()
            log_col = col.log()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        avg_time = sum(times) / rounds
        log_times.append(avg_time)
        print(f"log - Batch size: {batch_size}, Average Time over {rounds} runs: {avg_time:.3f} milliseconds")
    fig = go.Figure(data=go.Scatter(x=batch_sizes, y=log_times, mode='lines+markers'))
    fig.update_layout(title='Average Processing Time vs Batch Size for log',
                      xaxis_title='Batch Size',
                      yaxis_title='Average Processing Time (milliseconds)')
    fig.write_image('log_performance.png')


def test_continuous(batch_sizes, rounds=5):
    continuous_times = []
    for batch_size in batch_sizes:
        times = []
        for _ in range(rounds):
            # Generate continuous data between 0 and 100 as float32
            data = (np.random.rand(batch_size) * 100).astype(np.float32)
            col = ta.column(data, dtype=ta.float32)
            # Define bucket borders as float32
            borders = [20.0, 40.0, 60.0, 80.0]
            borders = [np.float32(b) for b in borders]
            start_time = time.time()
            # Bucketize the data
            bucketized_col = functional.bucketize(col, borders)
            # Hash the bucketized values
            salt = 0
            max_value = 100
            hashed_col = functional.sigrid_hash(
                bucketized_col, salt=salt, max_value=max_value
            )
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        avg_time = sum(times) / rounds
        continuous_times.append(avg_time)
        print(
            f"Continuous (Bucketize + Hash) - Batch size: {batch_size}, Average Time over {rounds} runs: {avg_time:.3f} milliseconds"
        )
    fig = go.Figure(
        data=go.Scatter(x=batch_sizes, y=continuous_times, mode="lines+markers")
    )
    fig.update_layout(
        title="Average Processing Time vs Batch Size for Continuous Data (Bucketize + Hash)",
        xaxis_title="Batch Size",
        yaxis_title="Average Processing Time (milliseconds)",
    )
    fig.write_image("continuous_performance.png")


batch_sizes = [1000, 10000, 100000, 500000, 1000000]
# test_sigrid_hash(batch_sizes, rounds=50)
# test_bucketize(batch_sizes, rounds=50)
# test_log(batch_sizes, rounds=50)
test_continuous(batch_sizes, rounds=50)
