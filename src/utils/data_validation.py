import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd


def build_suite(include_target: bool) -> gx.ExpectationSuite:
    """
    Build the expectation suite: the list of rules the data must satisfy.

    Args:
        include_target (bool): Add the Yield rule (only the train data has Yield).

    Returns:
        gx.ExpectationSuite: The suite of expectations.
    """
    suite = gx.ExpectationSuite(name="crop_data_suite")

    # land cannot be zero: you cannot predict yield for land that is not under cultivation
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="CultLand", min_value=0, strict_min=True))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="CropCultLand", min_value=0, strict_min=True))

    # seedlings per pit cannot be zero: something must have been planted
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="SeedlingsPerPit", min_value=0, strict_min=True))

    # yield cannot be negative (zero is allowed, e.g. a failed crop)
    if include_target:
        suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="Yield", min_value=0))

    return suite


def validate_data(df: pd.DataFrame, name: str = "data") -> bool:
    """
    Validate a raw DataFrame against the expectation suite and print any failed rules.

    Args:
        df (pd.DataFrame): The raw data to validate.
        name (str): A label for the data, used in the printed report.

    Returns:
        bool: True if every expectation passed, False otherwise.
    """
    # an ephemeral context keeps everything in memory, nothing is written to disk
    context = gx.get_context(mode="ephemeral")

    # tell GX where the data comes from: a pandas DataFrame passed in at run time
    data_source = context.data_sources.add_pandas(name="pandas")
    data_asset = data_source.add_dataframe_asset(name=name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name="whole_dataframe")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(build_suite(include_target="Yield" in df.columns))
    results = batch.validate(suite)

    # print a short report for each rule
    for res in results.results:
        column = res.expectation_config.kwargs["column"]
        status = "PASS" if res.success else "FAIL"
        unexpected = res.result.get("unexpected_count", 0)
        print(f"[{status}] {name}.{column}: {unexpected} bad values")
        if not res.success:
            print(f"        examples: {res.result.get('partial_unexpected_list')}")

    return results.success
