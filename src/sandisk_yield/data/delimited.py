"""CSV/TSV reader; file extension does not determine the delimiter."""
import csv
import logging
import pandas as pd


def read_delimited(path, separator=None):
    if separator is None:
        # Inspect only the header when the loader is invoked. Restrict candidates
        # so spaces inside the block_readings sequence cannot become delimiters.
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
        try:
            separator = csv.Sniffer().sniff(header, delimiters=",\t;").delimiter
        except csv.Error as exc:
            raise ValueError(f"Cannot detect delimiter in {path}. Expected a comma-, tab-, "
                             "or semicolon-separated header; specify separator explicitly in the loader.") from exc
    logging.getLogger("sandisk_yield.loader").info("Reading %s with delimiter %r", path, separator)
    return pd.read_csv(path, sep=separator, encoding="utf-8-sig", low_memory=False)
