import torcharrow as ta
from torcharrow import functional
import time
import torcharrow.dtypes as dt
import numpy as np
import plotly.graph_objects as go


def compute_batch_size(total_embedding_lookups, num_embeddings, lookups_per_embedding):
    return total_embedding_lookups // (num_embeddings * lookups_per_embedding)


def generate_synthetic_batch(
    batch_size, num_embeddings, lookups_per_embedding, data_type="int"
):
    data = []
    for _ in range(batch_size):
        sample = []
        for _ in range(num_embeddings):
            if data_type == "int":
                lookups = np.random.randint(1, 1000000, size=lookups_per_embedding)
            elif data_type == "float":
                lookups = np.random.rand(lookups_per_embedding) * 100 + 1
            else:
                raise ValueError("Unsupported data type")
            sample.extend(lookups)
        data.append(sample)
    flat_data = [item for sublist in data for item in sublist]
    col = ta.column(flat_data)
    return col


def test_sigrid_hash(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=5
):
    batch_size = compute_batch_size(
        total_embedding_lookups, num_embeddings, lookups_per_embedding
    )
    times = []
    for _ in range(rounds):
        col = generate_synthetic_batch(
            batch_size, num_embeddings, lookups_per_embedding, data_type="int"
        )
        salt = 0
        max_value = 100
        start_time = time.time()
        hashed_col = functional.sigrid_hash(col, salt=salt, max_value=max_value)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    avg_time = sum(times) / rounds
    print(
        f"sigrid_hash - Batch size: {batch_size}, Embeddings: {num_embeddings}, "
        f"Lookups per Embedding: {lookups_per_embedding}, "
        f"Average Time over {rounds} runs: {avg_time:.3f} milliseconds"
    )


def test_bucketize(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=5
):
    batch_size = compute_batch_size(
        total_embedding_lookups, num_embeddings, lookups_per_embedding
    )
    times = []
    for _ in range(rounds):
        col = generate_synthetic_batch(
            batch_size, num_embeddings, lookups_per_embedding, data_type="int"
        )
        borders = [2, 5, 10]
        start_time = time.time()
        bucketized_col = functional.bucketize(col, borders)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    avg_time = sum(times) / rounds
    print(
        f"bucketize - Batch size: {batch_size}, Embeddings: {num_embeddings}, "
        f"Lookups per Embedding: {lookups_per_embedding}, "
        f"Average Time over {rounds} runs: {avg_time:.3f} milliseconds"
    )


def test_log(total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=5):
    batch_size = compute_batch_size(
        total_embedding_lookups, num_embeddings, lookups_per_embedding
    )
    times = []
    for _ in range(rounds):
        col = generate_synthetic_batch(
            batch_size, num_embeddings, lookups_per_embedding, data_type="float"
        )
        start_time = time.time()
        log_col = col.log()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    avg_time = sum(times) / rounds
    print(
        f"log - Batch size: {batch_size}, Embeddings: {num_embeddings}, "
        f"Lookups per Embedding: {lookups_per_embedding}, "
        f"Average Time over {rounds} runs: {avg_time:.3f} milliseconds"
    )


def test_continuous(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=5
):
    batch_size = compute_batch_size(
        total_embedding_lookups, num_embeddings, lookups_per_embedding
    )
    times = []
    for _ in range(rounds):
        col = generate_synthetic_batch(
            batch_size, num_embeddings, lookups_per_embedding, data_type="int"
        )
        col = col.cast(dt.int64)
        borders = [20, 40, 60, 80]
        start_time = time.time()
        bucketized_col = functional.bucketize(col, borders)
        bucketized_col = bucketized_col.cast(dt.int64)
        salt = 0
        max_value = 100
        hashed_col = functional.sigrid_hash(
            bucketized_col, salt=salt, max_value=max_value
        )
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    avg_time = sum(times) / rounds
    print(
        f"Continuous (Bucketize + Hash) - Batch size: {batch_size}, Embeddings: {num_embeddings}, "
        f"Lookups per Embedding: {lookups_per_embedding}, "
        f"Average Time over {rounds} runs: {avg_time:.3f} milliseconds"
    )


total_embedding_lookups = 1_000_000

num_embeddings = 50

# Test Case 1: One-Hot Encoding (Lookups per Embedding = 1)
lookups_per_embedding = 1
print("Testing with One-Hot Encoding")
test_sigrid_hash(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=50
)
test_bucketize(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=50
)
test_log(total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=50)
test_continuous(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=50
)

# Test Case 2: Multi-Hot Encoding (Lookups per Embedding = 80)
lookups_per_embedding = 80
print("\nTesting with Multi-Hot Encoding")
test_sigrid_hash(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=50
)
test_bucketize(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=50
)
test_log(total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=50)
test_continuous(
    total_embedding_lookups, num_embeddings, lookups_per_embedding, rounds=50
)
