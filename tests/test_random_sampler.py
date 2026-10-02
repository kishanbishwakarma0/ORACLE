
from src.active_learning.random_sampler import RandomSampler


def main():

    # Simulate an unlabeled pool of 1000 samples
    pool = list(range(1000))

    sampler = RandomSampler(
        pool_indices=pool,
        seed=42,
    )

    print("--- Random Sampling Test ---")

    print(f"Initial pool size: {sampler.get_pool_size()}")

    # First query
    selected_1 = sampler.query(100)

    print(f"First query size: {len(selected_1)}")
    print(f"Remaining pool: {sampler.get_pool_size()}")

    # Second query
    selected_2 = sampler.query(100)

    print(f"Second query size: {len(selected_2)}")
    print(f"Remaining pool: {sampler.get_pool_size()}")

    # Verify no duplicate selections
    assert len(set(selected_1)) == 100
    assert len(set(selected_2)) == 100
    assert set(selected_1).isdisjoint(set(selected_2))

    # Verify correct pool size
    assert sampler.get_pool_size() == 800

    print("\nAll Random Sampling tests passed!")


if __name__ == "__main__":
    main()