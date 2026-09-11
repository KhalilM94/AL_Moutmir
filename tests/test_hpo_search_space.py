import optuna
import pytest

from yg_eo_soilnet.hpo.overrides import apply_overrides
from yg_eo_soilnet.hpo.search_space import (
    Distribution,
    Objective,
    SearchSpace,
    load_search_spaces_document,
)

SEARCH_SPACES_PATH = "configs/lightning/search_spaces"  # the folder tune.py defaults to


def _space(**overrides) -> SearchSpace:
    mapping = {"params": {"model.learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}}}
    mapping.update(overrides)
    return SearchSpace.from_mapping("fake_entry", mapping)


# --- objective ---------------------------------------------------------------


def test_objective_defaults_to_maximizing_val_r2():
    objective = Objective.from_mapping(None)

    assert (objective.metric, objective.direction, objective.mode) == ("val_r2", "maximize", "max")


def test_objective_mode_follows_the_direction():
    """EarlyStopping and Optuna must never disagree about which way is better."""
    assert Objective.from_mapping({"metric": "val_loss", "direction": "minimize"}).mode == "min"


def test_an_unknown_direction_is_rejected():
    with pytest.raises(ValueError, match="objective.direction"):
        Objective.from_mapping({"direction": "smaller"})


def test_a_metric_no_model_logs_is_rejected_at_load_time():
    """Otherwise every trial finishes, finds nothing to score, and is pruned - a whole study lost."""
    with pytest.raises(ValueError, match="is not logged by any model"):
        Objective.from_mapping({"metric": "mse", "direction": "minimize"})


def test_a_loss_shaped_typo_gets_the_val_loss_hint():
    with pytest.raises(ValueError, match="For mean squared error use 'val_loss'"):
        Objective.from_mapping({"metric": "mse", "direction": "minimize"})


@pytest.mark.parametrize("metric", ["val_loss", "val_r2", "val_pred_std_ratio", "test_r2"])
def test_every_logged_metric_is_accepted(metric):
    assert Objective.from_mapping({"metric": metric, "direction": "minimize"}).metric == metric


# --- distribution validation -------------------------------------------------


def test_log_and_step_together_are_rejected():
    with pytest.raises(ValueError, match="rejects 'log' and 'step' together"):
        Distribution.from_mapping("model.lr", {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "step": 0.1})


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError, match="is above high"):
        Distribution.from_mapping("model.lr", {"type": "float", "low": 1.0, "high": 0.1})


def test_missing_bounds_are_named():
    with pytest.raises(ValueError, match="needs low and high"):
        Distribution.from_mapping("model.lr", {"type": "float"})


def test_an_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="unknown type 'uniform'"):
        Distribution.from_mapping("model.lr", {"type": "uniform", "low": 0, "high": 1})


def test_a_list_valued_categorical_choice_is_rejected():
    """Optuna stores choices in the study DB; structured values belong in a derive hook."""
    with pytest.raises(ValueError, match="Use a 'derive' hook"):
        Distribution.from_mapping("model.head_hidden_dims", {"type": "categorical", "choices": [[64, 32], [32]]})


def test_an_empty_categorical_is_rejected():
    with pytest.raises(ValueError, match="non-empty 'choices'"):
        Distribution.from_mapping("model.activation", {"type": "categorical", "choices": []})


# --- space validation --------------------------------------------------------


def test_an_unknown_top_level_key_is_rejected():
    with pytest.raises(ValueError, match="unknown key\\(s\\): parms"):
        SearchSpace.from_mapping("fake_entry", {"parms": {}})


def test_a_space_with_nothing_to_search_is_rejected():
    with pytest.raises(ValueError, match="declares no 'params'"):
        SearchSpace.from_mapping("fake_entry", {"objective": {"metric": "val_r2"}})


def test_guarded_keys_are_caught_at_load_time_not_mid_study():
    with pytest.raises(ValueError, match="resolved from the datamodule"):
        SearchSpace.from_mapping("fake_entry", {"params": {"model.target_dim": {"type": "int", "low": 1, "high": 4}}})


def test_a_forward_when_reference_is_rejected():
    """Draw order is declaration order, so a forward guard would silently never match."""
    mapping = {
        "params": {
            "model.nhead": {"type": "categorical", "choices": [2, 4], "when": {"model.temporal_encoder": "x"}},
            "model.temporal_encoder": {"type": "categorical", "choices": ["x", "y"]},
        }
    }
    with pytest.raises(ValueError, match="not\\s+declared before it"):
        SearchSpace.from_mapping("fake_entry", mapping)


def test_an_unknown_derive_hook_is_rejected():
    with pytest.raises(ValueError, match="Unknown constraint hook"):
        _space(derive=["no_such_hook"])


# --- suggestion --------------------------------------------------------------


def test_fixed_values_are_applied_to_every_trial():
    space = _space(fixed={"trainer.max_epochs": 150})
    chosen = space.suggest(optuna.trial.FixedTrial({"model.learning_rate": 0.003}))

    assert chosen["trainer.max_epochs"] == 150
    assert chosen["model.learning_rate"] == 0.003


def test_a_when_guard_suppresses_the_parameter_when_it_does_not_match():
    mapping = {
        "params": {
            "model.temporal_encoder": {"type": "categorical", "choices": ["time_transformer", "time_lstm"]},
            "model.nhead": {"type": "categorical", "choices": [2, 4], "when": {"model.temporal_encoder": "time_transformer"}},
        }
    }
    space = SearchSpace.from_mapping("fake_entry", mapping)

    transformer = space.suggest(optuna.trial.FixedTrial({"model.temporal_encoder": "time_transformer", "model.nhead": 4}))
    lstm = space.suggest(optuna.trial.FixedTrial({"model.temporal_encoder": "time_lstm"}))

    assert transformer["model.nhead"] == 4
    assert "model.nhead" not in lstm


def test_derive_hook_repairs_d_model_to_divide_by_nhead():
    """TimeAwareTransformerEncoder raises unless d_model % nhead == 0."""
    mapping = {
        "params": {
            "model.nhead": {"type": "categorical", "choices": [8]},
            "model.d_model": {"type": "categorical", "choices": [50]},
        },
        "derive": ["d_model_divisible_by_nhead"],
    }
    space = SearchSpace.from_mapping("fake_entry", mapping)
    chosen = space.suggest(optuna.trial.FixedTrial({"model.nhead": 8, "model.d_model": 50}))

    assert chosen["model.d_model"] == 56
    assert chosen["model.d_model"] % chosen["model.nhead"] == 0


def _attention_pair_space(derive) -> SearchSpace:
    mapping = {
        "params": {
            "model.attention_nhead": {"type": "categorical", "choices": [8]},
            "model.attention_d_model": {"type": "categorical", "choices": [50]},
        },
        "derive": derive,
    }
    return SearchSpace.from_mapping("fake_entry", mapping)


def test_the_divisibility_hook_repairs_whichever_pair_it_is_pointed_at():
    """The residual attention CNN names its own pair; the sequence transformer keeps the default."""
    space = _attention_pair_space(
        [
            {
                "d_model_divisible_by_nhead": {
                    "d_model_key": "model.attention_d_model",
                    "nhead_key": "model.attention_nhead",
                }
            }
        ]
    )
    chosen = space.suggest(optuna.trial.FixedTrial({"model.attention_nhead": 8, "model.attention_d_model": 50}))

    assert chosen["model.attention_d_model"] == 56
    assert "model.d_model" not in chosen


def test_the_bare_divisibility_hook_leaves_other_pairs_alone():
    space = _attention_pair_space(["d_model_divisible_by_nhead"])
    chosen = space.suggest(optuna.trial.FixedTrial({"model.attention_nhead": 8, "model.attention_d_model": 50}))

    assert chosen["model.attention_d_model"] == 50


def test_derive_hook_builds_a_head_pyramid_of_plain_ints():
    space = _space(derive=["dims_pyramid"])
    chosen = space.suggest(
        optuna.trial.FixedTrial({"model.learning_rate": 0.001, "head_hidden_dims_depth": 3, "head_hidden_dims_width": 128})
    )

    dims = chosen["model.head_hidden_dims"]
    assert dims == [128, 64, 32]
    assert all(type(dim) is int for dim in dims)


def test_suggested_values_reach_the_right_registry_sections():
    """The end-to-end contract: a drawn trial becomes a runnable registry entry."""
    space = SearchSpace.from_mapping(
        "fake_entry",
        {
            "fixed": {"trainer.max_epochs": 150},
            "params": {
                "model.dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
                "datamodule.batch_size": {"type": "categorical", "choices": [16, 64]},
            },
        },
    )
    chosen = space.suggest(optuna.trial.FixedTrial({"model.dropout": 0.25, "datamodule.batch_size": 64}))
    spec = apply_overrides({"enabled": True, "init_args": {"static_dim": "auto"}}, chosen)

    assert spec["init_args"] == {"static_dim": "auto", "dropout": 0.25}
    assert spec["datamodule_init_args"] == {"batch_size": 64}
    assert spec["trainer_args"] == {"max_epochs": 150}


# --- samplers, pruners, and the shipped file ---------------------------------


def test_sampler_and_pruner_are_built_from_the_spec():
    space = _space(sampler={"name": "random", "seed": 7}, pruner={"name": "none"})

    assert isinstance(space.make_sampler(), optuna.samplers.RandomSampler)
    assert isinstance(space.make_pruner(), optuna.pruners.NopPruner)


def test_defaults_are_tpe_and_median():
    space = _space()

    assert isinstance(space.make_sampler(), optuna.samplers.TPESampler)
    assert isinstance(space.make_pruner(), optuna.pruners.MedianPruner)


SHIPPED_ENTRIES = ["soil_tabular", "soil_sequence", "soil_cnn", "soil_residual_cnn", "soil_residual_attention_cnn"]


@pytest.mark.parametrize("entry", SHIPPED_ENTRIES)
def test_the_shipped_search_spaces_load_and_draw(entry):
    """Every shipped space must survive validation and produce a full random draw."""
    space = SearchSpace.from_yaml(SEARCH_SPACES_PATH, entry)
    study = optuna.create_study(direction=space.objective.direction, sampler=space.make_sampler())
    chosen = space.suggest(study.ask())

    assert chosen
    assert space.describe()
    # Nothing drawn may collide with a guarded key, and everything must be routable.
    apply_overrides({"enabled": True}, chosen)


def test_a_missing_entry_names_what_is_available():
    with pytest.raises(KeyError, match="soil_tabular"):
        SearchSpace.from_yaml(SEARCH_SPACES_PATH, "no_such_model")


# --- splitting entries into a sibling folder ---------------------------------


def test_entries_can_be_split_into_a_sibling_folder(tmp_path):
    """configs/lightning/search_spaces/*.yml works the same as writing entries inline."""
    main_path = tmp_path / "search_spaces.yml"
    main_path.write_text("# just the shared header, no entries\n")
    split_dir = tmp_path / "search_spaces"
    split_dir.mkdir()
    (split_dir / "toy_model.yml").write_text(
        "toy_model:\n  params:\n    model.learning_rate: {type: float, low: 1.0e-4, high: 1.0e-2}\n"
    )

    space = SearchSpace.from_yaml(str(main_path), "toy_model")

    assert space.entry == "toy_model"
    assert space.distributions


def test_a_folder_can_be_pointed_at_directly(tmp_path):
    """`tune.py --search-spaces <folder>`: no index file involved at all."""
    split_dir = tmp_path / "search_spaces"
    split_dir.mkdir()
    (split_dir / "toy_model.yml").write_text(
        "toy_model:\n  params:\n    model.learning_rate: {type: float, low: 1.0e-4, high: 1.0e-2}\n"
    )

    space = SearchSpace.from_yaml(str(split_dir), "toy_model")

    assert space.entry == "toy_model"
    assert space.distributions


def test_naming_a_file_that_does_not_exist_still_finds_the_folder(tmp_path):
    """The folder is derived from the path as a string, so the file it is named after may be gone.

    That is the layout on disk: configs/lightning/search_spaces.yml was deleted once every entry
    had moved into configs/lightning/search_spaces/.
    """
    split_dir = tmp_path / "search_spaces"
    split_dir.mkdir()
    (split_dir / "toy_model.yml").write_text(
        "toy_model:\n  params:\n    model.dropout: {type: float, low: 0.0, high: 0.5}\n"
    )

    document = load_search_spaces_document(str(tmp_path / "search_spaces.yml"))

    assert set(document) == {"toy_model"}


def test_the_folder_and_the_file_naming_it_load_the_same_spaces(tmp_path):
    """Both spellings must fingerprint alike, or switching them would fork the Optuna study."""
    split_dir = tmp_path / "search_spaces"
    split_dir.mkdir()
    (split_dir / "toy_model.yml").write_text(
        "toy_model:\n  params:\n    model.learning_rate: {type: float, low: 1.0e-4, high: 1.0e-2}\n"
    )

    by_folder = SearchSpace.from_yaml(str(split_dir), "toy_model")
    by_filename = SearchSpace.from_yaml(str(tmp_path / "search_spaces.yml"), "toy_model")

    assert by_folder.fingerprint() == by_filename.fingerprint()


def test_split_entries_merge_alongside_inline_entries(tmp_path):
    """A search-space file can mix entries still written inline with ones split into their own file."""
    main_path = tmp_path / "search_spaces.yml"
    main_path.write_text(
        "inline_model:\n  params:\n    model.learning_rate: {type: float, low: 1.0e-4, high: 1.0e-2}\n"
    )
    split_dir = tmp_path / "search_spaces"
    split_dir.mkdir()
    (split_dir / "split_model.yml").write_text(
        "split_model:\n  params:\n    model.dropout: {type: float, low: 0.0, high: 0.5}\n"
    )

    document = load_search_spaces_document(str(main_path))

    assert set(document) == {"inline_model", "split_model"}


def test_a_path_with_no_sibling_folder_is_unaffected(tmp_path):
    """No search_spaces/ folder next to the file (the common case) is not an error."""
    main_path = tmp_path / "search_spaces.yml"
    main_path.write_text(
        "solo:\n  params:\n    model.learning_rate: {type: float, low: 1.0e-4, high: 1.0e-2}\n"
    )

    document = load_search_spaces_document(str(main_path))

    assert set(document) == {"solo"}


def test_duplicate_entry_across_main_file_and_split_folder_raises(tmp_path):
    """The same entry name declared both inline and in the split folder is a config mistake."""
    main_path = tmp_path / "search_spaces.yml"
    main_path.write_text(
        "toy_model:\n  params:\n    model.learning_rate: {type: float, low: 1.0e-4, high: 1.0e-2}\n"
    )
    split_dir = tmp_path / "search_spaces"
    split_dir.mkdir()
    (split_dir / "toy_model.yml").write_text(
        "toy_model:\n  params:\n    model.dropout: {type: float, low: 0.0, high: 0.5}\n"
    )

    with pytest.raises(ValueError, match="toy_model"):
        load_search_spaces_document(str(main_path))


# --- derive hook options -----------------------------------------------------


def test_a_derive_hook_takes_options_from_the_mapping_form():
    """One hook, different floors per model - the sequence head floors at 16, the CNN's at 8."""
    space = _space(derive=[{"dims_pyramid": {"widths": [64], "floor": 16, "max_depth": 4}}])
    chosen = space.suggest(
        optuna.trial.FixedTrial({"model.learning_rate": 0.001, "head_hidden_dims_depth": 4, "head_hidden_dims_width": 64})
    )

    assert chosen["model.head_hidden_dims"] == [64, 32, 16, 16]  # unfloored this would end 16 -> 8


def test_the_pyramid_depth_range_is_configurable():
    space = _space(derive=[{"dims_pyramid": {"min_depth": 2, "max_depth": 2}}])
    trial = optuna.create_study().ask()
    space.suggest(trial)

    assert trial.params["head_hidden_dims_depth"] == 2


def test_the_bare_name_and_the_mapping_form_are_both_accepted():
    bare = _space(derive=["dims_pyramid"])
    mapped = _space(derive=[{"dims_pyramid": {}}])
    params = {"model.learning_rate": 0.001, "head_hidden_dims_depth": 2, "head_hidden_dims_width": 64}

    assert bare.suggest(optuna.trial.FixedTrial(dict(params))) == mapped.suggest(
        optuna.trial.FixedTrial(dict(params))
    )


@pytest.mark.parametrize("entry", [["a", "b"], [{"one": {}, "two": {}}], [42], [None]])
def test_a_malformed_derive_entry_is_refused_at_load_time(entry):
    with pytest.raises(ValueError, match="Malformed 'derive' entry|Unknown constraint hook"):
        _space(derive=entry)


def test_an_unknown_option_fails_at_draw_time_naming_the_hook():
    space = _space(derive=[{"dims_pyramid": {"no_such_option": 1}}])

    with pytest.raises(TypeError, match="no_such_option"):
        space.suggest(optuna.trial.FixedTrial({"model.learning_rate": 0.001}))


def test_describe_renders_a_hook_with_its_options():
    space = _space(derive=[{"dims_pyramid": {"floor": 16, "max_depth": 2}}])

    assert space.describe()["space.derive"] == "dims_pyramid(floor=16, max_depth=2)"


# --- fingerprint -------------------------------------------------------------


def test_the_same_space_fingerprints_the_same_twice():
    assert _space().fingerprint() == _space().fingerprint()


@pytest.mark.parametrize(
    "overrides",
    [
        {"objective": {"metric": "val_loss", "direction": "minimize"}},
        {"objective": {"metric": "val_pred_std_ratio", "direction": "maximize"}},  # same direction
        {"fixed": {"trainer.max_epochs": 150}},
        {"derive": ["dims_pyramid"]},
        {"derive": [{"dims_pyramid": {"floor": 16}}]},
        {"params": {"model.learning_rate": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True}}},
        {"params": {"model.dropout": {"type": "float", "low": 0.0, "high": 0.5}}},
    ],
)
def test_editing_the_space_changes_the_fingerprint(overrides):
    """Anything that makes two trials incomparable must move the digest, not just `direction`."""
    assert _space(**overrides).fingerprint() != _space().fingerprint()


@pytest.mark.parametrize(
    "overrides",
    [{"sampler": {"name": "random", "seed": 7}}, {"pruner": {"name": "none"}}],
)
def test_the_sampler_and_the_pruner_are_not_part_of_the_fingerprint(overrides):
    """They change HOW the space is searched, not what a recorded value means, so resuming is fine."""
    assert _space(**overrides).fingerprint() == _space().fingerprint()


def test_the_shipped_spaces_fingerprint_distinctly():
    digests = {SearchSpace.from_yaml(SEARCH_SPACES_PATH, entry).fingerprint() for entry in SHIPPED_ENTRIES}

    assert len(digests) == len(SHIPPED_ENTRIES)


# --- dims_pyramid targets any list-valued key --------------------------------


def test_the_pyramid_writes_the_key_it_is_given():
    space = _space(derive=[{"dims_pyramid": {"key": "model.cnn_hidden_dims", "widths": [64]}}])
    chosen = space.suggest(
        optuna.trial.FixedTrial(
            {"model.learning_rate": 0.001, "cnn_hidden_dims_depth": 2, "cnn_hidden_dims_width": 64}
        )
    )

    assert chosen["model.cnn_hidden_dims"] == [64, 32]
    assert "model.head_hidden_dims" not in chosen


def test_two_pyramids_in_one_space_draw_distinct_parameters():
    """Parameter names come from the key; sharing them would make one stack shadow the other."""
    space = _space(
        derive=[
            {"dims_pyramid": {"key": "model.head_hidden_dims", "widths": [64]}},
            {"dims_pyramid": {"key": "model.static_hidden_dims", "widths": [32], "taper": 1.0}},
        ]
    )
    trial = optuna.create_study().ask()
    chosen = space.suggest(trial)

    assert {"head_hidden_dims_depth", "head_hidden_dims_width"} <= set(trial.params)
    assert {"static_hidden_dims_depth", "static_hidden_dims_width"} <= set(trial.params)
    assert chosen["model.head_hidden_dims"] and chosen["model.static_hidden_dims"]


@pytest.mark.parametrize(
    "taper,expected",
    [(0.5, [128, 64, 32]), (1.0, [128, 128, 128]), (2.0, [128, 256, 512])],
)
def test_the_taper_selects_the_shape(taper, expected):
    space = _space(derive=[{"dims_pyramid": {"widths": [128], "taper": taper, "floor": 1}}])
    chosen = space.suggest(
        optuna.trial.FixedTrial(
            {"model.learning_rate": 0.001, "head_hidden_dims_depth": 3, "head_hidden_dims_width": 128}
        )
    )

    assert chosen["model.head_hidden_dims"] == expected


def test_depth_zero_yields_an_empty_list_and_draws_no_width():
    """A bare readout was unreachable while min_depth floored at 1, and soil_sequence searched it."""
    space = _space(derive=[{"dims_pyramid": {"min_depth": 0, "max_depth": 0}}])
    trial = optuna.create_study().ask()
    chosen = space.suggest(trial)

    assert chosen["model.head_hidden_dims"] == []
    assert "head_hidden_dims_width" not in trial.params  # no dimension that changes nothing


def test_the_shipped_sequence_space_can_still_reach_a_bare_readout():
    space = SearchSpace.from_yaml(SEARCH_SPACES_PATH, "soil_sequence")
    depths = {
        entry["dims_pyramid"]["min_depth"]
        for entry in space.derive
        if isinstance(entry, dict) and entry["dims_pyramid"]["key"] == "model.head_hidden_dims"
    }

    assert depths == {0}
