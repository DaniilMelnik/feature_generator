from pathlib import Path

import click
import pandas as pd
import pytest
from click.testing import CliRunner

from feature_generator.cli import build_dependencies, main, make_feature_computer_factory
from feature_generator.config import DatasetConfig, RunConfig, TableConfig
from feature_generator.sandbox.docker_runner import SandboxedFeatureComputer
from feature_generator.schemas import FeatureSpec


def _config(tmp_path: Path) -> RunConfig:
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"Age": list(range(10)), "Transported": [True, False] * 5}).to_csv(csv_path, index=False)
    return RunConfig(
        dataset=DatasetConfig(
            name="mini",
            tables=[TableConfig(name="passengers", path=str(csv_path), role="primary")],
            target_column="Transported",
            target_table="passengers",
        ),
        cv_folds=2,
    )


def test_build_dependencies_raises_clear_error_without_docker(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("feature_generator.cli.docker_available", lambda: False)

    with pytest.raises(click.ClickException, match="Docker daemon is not reachable"):
        build_dependencies(config)


def test_build_dependencies_wires_real_components_when_docker_available(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    config.output.run_dir = str(tmp_path / "runs")
    config.output.knowledge_base_path = str(tmp_path / "runs" / "kb.duckdb")
    config.output.results_store_path = str(tmp_path / "runs" / "results.duckdb")
    config.output.feature_store_dir = str(tmp_path / "runs" / "feature_store")
    monkeypatch.setattr("feature_generator.cli.docker_available", lambda: True)
    monkeypatch.setattr("feature_generator.cli.LLMClient", lambda: object())

    deps = build_dependencies(config)

    assert deps.base_df.shape[0] == 10
    assert list(deps.y) == [1, 0] * 5
    assert deps.data_profile.dataset_name == "mini"
    assert deps.hypothesis_tier.model == config.model_tiers.hypothesis.model
    deps.knowledge_base.close()
    deps.results_store.close()


def test_make_feature_computer_factory_produces_sandboxed_computer(tmp_path: Path) -> None:
    config = _config(tmp_path)
    factory = make_feature_computer_factory(config)
    spec = FeatureSpec(
        hypothesis_id="h1", feature_name="f", module_source="x", declared_input_columns=["Age"],
        declared_input_tables=["passengers"], output_dtype="float64", codegen_model="claude-sonnet-5",
    )

    computer = factory(spec, True)

    assert isinstance(computer, SandboxedFeatureComputer)
    assert computer.feature_name == "f"
    assert computer.required_columns == ["Age"]
    assert computer.allow_target_in_fit is True


def test_cli_run_reports_missing_docker_as_a_clean_error(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"Age": [1, 2], "Transported": [True, False]}).to_csv(csv_path, index=False)
    config_path.write_text(
        f"""
dataset:
  name: mini
  tables:
    - name: passengers
      path: {csv_path}
      role: primary
  target_column: Transported
  target_table: passengers
"""
    )
    monkeypatch.setattr("feature_generator.cli.docker_available", lambda: False)

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "Docker daemon is not reachable" in result.output


def test_cli_help_lists_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "inspect" in result.output
