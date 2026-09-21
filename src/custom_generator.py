
import pandas as pd
import numpy as np
from faker import Faker
from datetime import date

fake = Faker()


def generate_custom_dataset(schema, num_rows, seed=None):
    """
    Generate a synthetic tabular dataset from a user-defined schema.

    schema: list of column configuration dictionaries
    num_rows: number of records
    seed: optional random seed
    """

    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)

    if not schema:
        raise ValueError("Please add at least one column.")

    if num_rows < 1:
        raise ValueError("Number of rows must be greater than zero.")

    column_names = [column["name"].strip() for column in schema]

    if any(not name for name in column_names):
        raise ValueError("Column names cannot be empty.")

    if len(column_names) != len(set(column_names)):
        raise ValueError("Column names must be unique.")

    result = {}

    for column in schema:
        name = column["name"].strip()
        data_type = column["type"]

        if data_type == "Integer":
            minimum = int(column.get("min", 0))
            maximum = int(column.get("max", 100))

            if minimum > maximum:
                raise ValueError(
                    f"Invalid range for {name}."
                )

            result[name] = np.random.randint(
                minimum,
                maximum + 1,
                size=num_rows
            )

        elif data_type == "Decimal":
            minimum = float(column.get("min", 0))
            maximum = float(column.get("max", 100))

            if minimum > maximum:
                raise ValueError(
                    f"Invalid range for {name}."
                )

            result[name] = np.round(
                np.random.uniform(
                    minimum,
                    maximum,
                    size=num_rows
                ),
                2
            )

        elif data_type == "Category":
            categories = column.get("categories", [])

            if not categories:
                raise ValueError(
                    f"Add at least one category for {name}."
                )

            result[name] = np.random.choice(
                categories,
                size=num_rows
            )

        elif data_type == "Boolean":
            result[name] = np.random.choice(
                [True, False],
                size=num_rows
            )

        elif data_type == "Name":
            result[name] = [
                fake.name() for _ in range(num_rows)
            ]

        elif data_type == "Email":
            result[name] = [
                fake.email() for _ in range(num_rows)
            ]

        elif data_type == "Date":
            start_date = column.get(
                "start_date", "2020-01-01"
            )
            end_date = column.get(
                "end_date", "2025-12-31"
            )

            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)

            if start > end:
                raise ValueError(
                    f"Invalid date range for {name}."
                )

            days = (end - start).days

            result[name] = [
                (
                    start + pd.Timedelta(
                        days=int(np.random.randint(0, days + 1))
                    )
                ).date()
                for _ in range(num_rows)
            ]

        elif data_type == "Unique ID":
            result[name] = [
                f"ID{index + 1:06d}"
                for index in range(num_rows)
            ]

        else:
            raise ValueError(
                f"Unsupported data type: {data_type}"
            )

    return pd.DataFrame(result)