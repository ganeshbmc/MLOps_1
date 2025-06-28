from feast import FileSource, FeatureView, Field
from feast.types import Float32
from entity import flower

sepal_source = FileSource(
    path="../data/iris_sepal.parquet",
    timestamp_field="event_timestamp",

)

petal_source = FileSource(
    path="../data/iris_petal.parquet",
    timestamp_field="event_timestamp",
)

# Define two different feature views (one each for sepal and petal features)
sepal_features = FeatureView(
    name="sepal_features",
    entities=[flower],
    ttl=None,
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
    ],
    source=sepal_source
)

petal_features = FeatureView(
    name="petal_features",
    entities=[flower],
    ttl=None,
    schema=[
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=petal_source
)
