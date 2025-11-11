import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession
from scipy.stats import chisquare

from qikits.sampling.weighted_sampling import weighted_sample


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("weighted_sampling_test")
        .getOrCreate()
    )


@pytest.fixture(scope="function")
def sample_df(spark):
    """Create a sample DataFrame for testing."""
    data = [
        (1, "A", 10.0),
        (2, "B", 20.0),
        (3, "C", 30.0),
        (4, "D", 40.0),
        (5, "E", 50.0),
    ]
    return spark.createDataFrame(data, ["id", "name", "weight"])


def test_basic_weighted_sampling(sample_df):
    """Test basic weighted sampling functionality."""
    k = 3
    result = weighted_sample(df=sample_df, k=k, weight_col="weight", id_columns="id")

    # Check the number of sampled rows
    assert result.count() == k

    # Check that all returned rows are from original DataFrame
    original_ids = {row.id for row in sample_df.select("id").collect()}
    sampled_ids = {row.id for row in result.select("id").collect()}
    assert sampled_ids.issubset(original_ids)


def test_deterministic_behavior(sample_df):
    """Test that sampling is deterministic with the same seed."""
    k = 3
    seed = "test_seed"

    # Run sampling twice with the same seed
    result1 = weighted_sample(
        df=sample_df, k=k, weight_col="weight", id_columns="id", seed=seed
    )
    result2 = weighted_sample(
        df=sample_df, k=k, weight_col="weight", id_columns="id", seed=seed
    )

    # Check that both results have the same IDs in the same order
    ids1 = [row.id for row in result1.select("id").collect()]
    ids2 = [row.id for row in result2.select("id").collect()]
    assert ids1 == ids2


def test_invalid_k(sample_df):
    """Test that invalid k values raise ValueError."""
    invalid_k_values = [0, -1, 1.5]

    for k in invalid_k_values:
        with pytest.raises(ValueError) as exc_info:
            weighted_sample(df=sample_df, k=k, weight_col="weight", id_columns="id")
        assert str(k) in str(exc_info.value)


def test_column_inputs(sample_df):
    """Test that both string and Column inputs work for weight_col and id_columns."""
    k = 2

    # Test with string inputs
    result1 = weighted_sample(df=sample_df, k=k, weight_col="weight", id_columns="id")

    # Test with Column inputs
    result2 = weighted_sample(
        df=sample_df, k=k, weight_col=F.col("weight"), id_columns=F.col("id")
    )

    # Both should return k rows
    assert result1.count() == k
    assert result2.count() == k


def test_multiple_id_columns(spark):
    """Test sampling with multiple ID columns."""
    # Create DataFrame with composite key
    data = [(1, "A", 10.0), (1, "B", 20.0), (2, "A", 30.0), (2, "B", 40.0)]
    df = spark.createDataFrame(data, ["group_id", "subgroup", "weight"])

    k = 2
    result = weighted_sample(
        df=df, k=k, weight_col="weight", id_columns=["group_id", "subgroup"]
    )

    # Check number of results
    assert result.count() == k

    # Check that composite keys are preserved
    assert result.select("group_id", "subgroup").distinct().count() == k


def test_proportional_sampling(spark):
    """Test that sampling probabilities are proportional to weights."""
    # Create data with known weights
    data = [
        (1, 10.0),  # Weight ratio 1:2:3:4
        (2, 20.0),
        (3, 30.0),
        (4, 40.0),
    ]
    df = spark.createDataFrame(data, ["id", "weight"]).cache()
    df.count()  # Trigger caching
    total_weight = sum(w for _, w in data)
    expected_probs = [w / total_weight for _, w in data]

    # Perform multiple samplings to get frequency distribution
    n_trials = 1000
    k = 1  # Sample one item at a time to maintain independence
    sample_counts = {id_: 0 for id_, _ in data}

    for i in range(n_trials):
        seed = f"trial_{i}"  # Different seed for each trial
        result = weighted_sample(
            df=df, k=k, weight_col="weight", id_columns="id", seed=seed
        )
        sampled_id = result.select("id").take(1)[0].id
        sample_counts[sampled_id] += 1

    # Convert counts to observed probabilities
    observed_counts = [sample_counts[id_] for id_, _ in data]
    observed_probs = [count / n_trials for count in observed_counts]

    # Expected counts under null hypothesis
    expected_counts = [n_trials * p for p in expected_probs]

    # Perform chi-square goodness of fit test
    chi2_stat, p_value = chisquare(observed_counts, expected_counts)

    # For the test to pass:
    # 1. p-value should be > 0.05 (assuming 5% significance level)
    # 2. Observed probabilities should be close to expected
    assert p_value > 0.05, (
        f"Chi-square test failed (p={p_value:.4f}). "
        f"Observed probabilities: {observed_probs}, "
        f"Expected probabilities: {expected_probs}"
    )

    # Additional check: observed probabilities should be within 10% of expected
    for obs, exp in zip(observed_probs, expected_probs):
        rel_diff = abs(obs - exp) / exp
        assert rel_diff < 0.1, (
            f"Observed probability {obs:.3f} differs from "
            f"expected {exp:.3f} by more than 10%"
        )
