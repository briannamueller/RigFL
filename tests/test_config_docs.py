"""Checks for the generated configuration reference."""

from scripts.generate_config_reference import missing_descriptions


def test_rigfl_configuration_fields_have_descriptions():
    assert missing_descriptions() == []
