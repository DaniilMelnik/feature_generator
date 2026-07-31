from pathlib import Path

import pytest
import yaml

from feature_generator.config import (
    DatasetConfig,
    LLMProviderConfig,
    ModelTierEntry,
    RunConfig,
    TableConfig,
    load_config,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize(
    "config_name",
    [
        "spaceship_titanic.yaml",
        "ieee_fraud.yaml",
        "spaceship_titanic_groq.yaml",
        "spaceship_titanic_gemini.yaml",
        "spaceship_titanic_ollama.yaml",
    ],
)
def test_shipped_configs_load_and_round_trip(config_name: str) -> None:
    cfg = load_config(CONFIGS_DIR / config_name)
    assert isinstance(cfg, RunConfig)

    # round-trip through JSON to catch any non-serializable defaults
    dumped = cfg.model_dump(mode="json")
    reloaded = RunConfig.model_validate(dumped)
    assert reloaded == cfg


def test_default_model_tiers_are_config_driven_not_hardcoded() -> None:
    cfg = load_config(CONFIGS_DIR / "spaceship_titanic.yaml")
    assert cfg.model_tiers.codegen.model == "claude-sonnet-5"
    assert cfg.model_tiers.leakage_reviewer.model == "claude-opus-5"
    assert cfg.model_tiers.utility.model == "claude-haiku-4-5"
    # escalation ladder is present and distinct from the base tier
    assert cfg.model_tiers.hypothesis.escalate_to_model == "claude-opus-5"
    assert cfg.model_tiers.hypothesis.escalate_to_model != cfg.model_tiers.hypothesis.model


def test_dataset_config_requires_exactly_one_primary_table() -> None:
    with pytest.raises(ValueError, match="exactly one primary table"):
        DatasetConfig(
            name="bad",
            tables=[
                TableConfig(name="a", path="a.csv", role="primary"),
                TableConfig(name="b", path="b.csv", role="primary"),
            ],
            target_column="y",
            target_table="a",
        )

    with pytest.raises(ValueError, match="exactly one primary table"):
        DatasetConfig(
            name="bad2",
            tables=[TableConfig(name="a", path="a.csv", role="secondary")],
            target_column="y",
            target_table="a",
        )


def test_ieee_fraud_config_has_temporal_settings() -> None:
    cfg = load_config(CONFIGS_DIR / "ieee_fraud.yaml")
    assert cfg.dataset.time_column == "TransactionDT"
    assert cfg.stability.method == "temporal_oot"
    secondary = [t for t in cfg.dataset.tables if t.role == "secondary"]
    assert len(secondary) == 1
    assert secondary[0].join_key == "TransactionID"


def test_spaceship_titanic_config_has_no_time_axis() -> None:
    cfg = load_config(CONFIGS_DIR / "spaceship_titanic.yaml")
    assert cfg.dataset.time_column is None
    assert cfg.stability.method == "bootstrap_resampling"


def test_load_config_raises_on_missing_required_field(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump({"dataset": {"name": "x"}}))
    with pytest.raises(Exception):
        load_config(bad_path)


def test_llm_provider_defaults_to_anthropic() -> None:
    cfg = load_config(CONFIGS_DIR / "spaceship_titanic.yaml")
    assert cfg.llm_provider == LLMProviderConfig(backend="anthropic", base_url=None, api_key_env="GROQ_API_KEY")


def test_model_tier_entry_max_output_tokens_defaults_to_32000() -> None:
    assert ModelTierEntry(model="claude-sonnet-5").max_output_tokens == 32000


def test_groq_config_selects_openai_compatible_backend() -> None:
    cfg = load_config(CONFIGS_DIR / "spaceship_titanic_groq.yaml")
    assert cfg.llm_provider.backend == "openai_compatible"
    assert cfg.llm_provider.base_url == "https://api.groq.com/openai/v1"
    assert cfg.llm_provider.api_key_env == "GROQ_API_KEY"
    assert cfg.model_tiers.codegen.model == "openai/gpt-oss-120b"
    assert cfg.model_tiers.codegen.max_output_tokens == 4000
