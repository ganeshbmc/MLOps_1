from feast import Entity
from feast.value_type import ValueType

flower = Entity(
    name="flower_id",
    join_keys=["flower_id"],
    value_type=ValueType.INT64,
    description="Unique ID for each flower sample"
)
