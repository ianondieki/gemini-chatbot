"""The registry turns functions into tools - and shapes errors so a model can
recover from them."""

from __future__ import annotations

import pytest

from gemini_agent.registry import ToolRegistry, build_spec


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool(describe={"expression": "A maths expression"})
    def calculate(expression: str, precision: int = 2) -> str:
        """Work out a sum."""
        return f"{expression}@{precision}"

    @registry.tool(concurrent_safe=False, cacheable=False)
    def note(ctx, fact: str, tags: list[str]) -> str:
        """Save a fact."""
        return f"{ctx}:{fact}:{tags}"

    return registry


class TestSchemaGeneration:
    def test_types_come_from_annotations(self, registry):
        parameters = registry.get("calculate").declaration.parameters
        assert parameters.properties["expression"].type.name == "STRING"
        assert parameters.properties["precision"].type.name == "INTEGER"

    def test_only_parameters_without_defaults_are_required(self, registry):
        assert registry.get("calculate").declaration.parameters.required == ["expression"]

    def test_lists_declare_their_item_type(self, registry):
        parameters = registry.get("note").declaration.parameters
        assert parameters.properties["tags"].type.name == "ARRAY"
        assert parameters.properties["tags"].items.type.name == "STRING"

    def test_context_is_hidden_from_the_model(self, registry):
        """``ctx`` is plumbing; the model must never see it as a parameter."""
        parameters = registry.get("note").declaration.parameters
        assert "ctx" not in parameters.properties
        assert registry.get("note").wants_context is True

    def test_description_falls_back_to_the_docstring(self, registry):
        assert registry.get("calculate").description == "Work out a sum."

    def test_a_tool_with_no_description_is_rejected(self):
        def bare(value: str) -> str:
            return value

        with pytest.raises(ValueError, match="description"):
            build_spec(bare)

    def test_varargs_are_rejected(self):
        def loose(*args) -> str:
            """Takes anything."""
            return ""

        with pytest.raises(ValueError, match=r"\*args"):
            build_spec(loose)

    def test_declarations_are_none_when_empty(self):
        assert ToolRegistry().declarations() is None


class TestDispatch:
    def test_calls_the_function(self, registry):
        assert registry.invoke("calculate", {"expression": "2+2"}).content == "2+2@2"

    def test_context_is_injected(self, registry):
        result = registry.invoke("note", {"fact": "x", "tags": ["a"]}, ctx="CTX")
        assert result.content == "CTX:x:['a']"

    def test_numeric_strings_are_coerced(self, registry):
        """Models routinely send "2" for an integer; a round trip to fix that is waste."""
        result = registry.invoke("calculate", {"expression": "1", "precision": "5"})
        assert result.ok and result.content == "1@5"

    def test_a_bare_string_is_accepted_for_a_list(self, registry):
        result = registry.invoke("note", {"fact": "x", "tags": "solo"}, ctx="C")
        assert result.content == "C:x:['solo']"

    def test_none_arguments_are_dropped_not_passed_through(self, registry):
        result = registry.invoke("calculate", {"expression": "1", "precision": None})
        assert result.ok and result.content == "1@2"


class TestErrorsGuideTheModel:
    """Every failure must tell the model enough to fix its next call."""

    def test_unknown_tool_lists_the_real_ones(self, registry):
        result = registry.invoke("teleport", {})
        assert not result.ok
        assert "calculate" in result.content and "note" in result.content

    def test_missing_argument_names_it_and_the_signature(self, registry):
        result = registry.invoke("calculate", {})
        assert not result.ok
        assert "'expression'" in result.content
        assert "precision: integer = 2" in result.content

    def test_unexpected_argument_is_named(self, registry):
        result = registry.invoke("calculate", {"expression": "1", "colour": "red"})
        assert not result.ok and "'colour'" in result.content

    def test_wrong_type_is_explained(self, registry):
        result = registry.invoke("calculate", {"expression": "1", "precision": "loads"})
        assert not result.ok and "must be a number" in result.content

    def test_an_exploding_tool_is_reported_not_raised(self):
        registry = ToolRegistry()

        @registry.tool()
        def boom() -> str:
            """Always fails."""
            raise RuntimeError("kaboom")

        result = registry.invoke("boom", {})
        assert not result.ok and "kaboom" in result.content


class TestComposition:
    def test_subset_narrows_the_toolbox(self, registry):
        narrowed = registry.subset(["calculate", "nonexistent"])
        assert narrowed.names == ["calculate"]

    def test_catalogue_is_readable_by_a_planner(self, registry):
        catalogue = registry.catalogue()
        assert "calculate(expression: string, precision: integer = 2)" in catalogue
        assert "Work out a sum." in catalogue

    def test_flags_survive_registration(self, registry):
        assert registry.get("note").concurrent_safe is False
        assert registry.get("note").cacheable is False
        assert registry.get("calculate").cacheable is True
