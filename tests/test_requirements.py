from tablemind.reasoning.requirements import (
    TaskRequirementBuilder,
)
from tablemind.reasoning.task_request import (
    TaskRequest,
)


def make_request(quantity: int = 4) -> TaskRequest:
    return TaskRequest(
        raw_instruction=f"Set the table for {quantity}.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=quantity,
    )


def test_builder_creates_requirement_for_each_object_type():
    requirements = TaskRequirementBuilder().build(
        make_request()
    )

    assert requirements.object_types == (
        "plate",
        "glass",
    )


def test_four_people_requires_four_plates():
    requirements = TaskRequirementBuilder().build(
        make_request(4)
    )

    plate_requirement = requirements.requirement_for(
        "plate"
    )

    assert plate_requirement.required_quantity == 4


def test_four_people_requires_four_glasses():
    requirements = TaskRequirementBuilder().build(
        make_request(4)
    )

    glass_requirement = requirements.requirement_for(
        "glass"
    )

    assert glass_requirement.required_quantity == 4


def test_two_people_requires_two_of_each_object():
    requirements = TaskRequirementBuilder().build(
        make_request(2)
    )

    assert (
        requirements.requirement_for("plate").required_quantity
        == 2
    )

    assert (
        requirements.requirement_for("glass").required_quantity
        == 2
    )


def test_missing_object_type_raises_key_error():
    requirements = TaskRequirementBuilder().build(
        make_request(4)
    )

    try:
        requirements.requirement_for("fork")
    except KeyError as exc:
        assert "fork" in str(exc)
    else:
        raise AssertionError(
            "Expected KeyError for unknown object type."
        )