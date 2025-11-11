import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame

_MIN_SIGNED_LONG_VAL = -(2**63)


def weighted_sample(
    df: DataFrame,
    k: int,
    weight_col: str | Column,
    id_columns: str | Column | list[str | Column],
    seed: str = "weighed_sampling_without_replacement",
) -> DataFrame:
    """Performs a deterministic, scalable weighted sampling without replacement.

    This function implements a scalable, one-pass algorithm for weighted random
    sampling without replacement, as described by Efraimidis and Spirakis [1].
    To ensure determinism, the "random" number used in the key calculation
    is derived from hashing the row's unique identifier(s) with a fixed seed.

    The core logic assigns a score to each item and selects the top k items.
    The score for an item `i` with weight `w_i` is calculated as `u_i^(1/w_i)`,
    where `u_i` is a deterministically generated number in the range [0, 1).

    Args:
        df: The input PySpark DataFrame.
        k: The number of items to sample. Must be a positive integer.
        weight_col: The name of the column containing the sampling weights.
            Weights must be positive numbers.
        id_columns: The column or the list of columns that uniquely identify each row.
        seed (optional): An string seed to ensure reproducibility.

    Returns:
        DataFrame: A new DataFrame containing the k sampled rows, with the
            additional 'sampling_score' column used for selection.

    References:
        [1] Efraimidis, P. S., & Spirakis, P. G. (2006). Weighted random sampling
            with a reservoir. Information Processing Letters, 97(5), 181-185.
            DOI: 10.1016/j.ipl.2005.11.003
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, but got {k}.")

    id_columns = id_columns if isinstance(id_columns, list) else [id_columns]
    weight_col = F.col(weight_col) if isinstance(weight_col, str) else weight_col

    # Step 1: Generate a deterministic hash value for each row.
    # xxhash64 returns a signed 64-bit long.
    hash_col = F.xxhash64(F.lit(seed), *id_columns)

    # 2. Normalize the signed hash to a deterministic value 'u' in the [0, 1) range.
    #    Spark's xxhash64 returns a signed long [-2**63, 2**63 - 1].
    #    We map this to [-1, 1) by dividing by the max signed long value,
    #    then scale and shift to [0, 1) via `(x+1)/2`.
    u_col = ((hash_col / F.lit(_MIN_SIGNED_LONG_VAL)) * F.lit(-1) + 1.0) / 2.0

    # 3: Calculate the score. score = u^(1/w)
    #    The logarithm is used to prevent potential floating point precision issues
    #    with large weights: log(score) = (1/w) * log(u).
    #    Since log is monotonic, the order is preserved.
    log_score_col = F.log(u_col) / weight_col

    # 4. Assign the score and select the top k items.
    ranked_df = df.orderBy(log_score_col.desc())

    return ranked_df.limit(k)
