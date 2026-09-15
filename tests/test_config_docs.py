"""Checks for the generated configuration reference."""

from scripts.generate_config_reference import (
    missing_descriptions,
    render_data_reference,
)


def test_rigfl_configuration_fields_have_descriptions():
    assert missing_descriptions() == []


def test_constrained_list_types_are_written_in_plain_language():
    reference = render_data_reference()

    assert "list of positive integers | required | non-empty" in reference
    assert "number \\| list of positive numbers" in reference
