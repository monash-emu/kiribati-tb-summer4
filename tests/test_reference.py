"""The reference-run kit: arm settings, every kernel's checkpoint and resume, the launcher."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from kiribati_tb.paths import REPO_ROOT
from kiribati_tb.pipeline import (
    KERNEL_FIELDS,
    Checkpoint,
    DeadlineReached,
    PipelineConfig,
    StagedWarmup,
    StopCriteria,
    make_mcmc,
    sample_chunks,
)
from kiribati_tb.reference import DEVICES, REFERENCE_ARMS, reference_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import reference_run  # noqa: E402

N_PARAMS = 19


def test_arms_are_the_four_requested() -> None:
    assert set(REFERENCE_ARMS) == {"nuts_td8", "nuts_td5", "sa", "ess"}
    td8, td5 = REFERENCE_ARMS["nuts_td8"], REFERENCE_ARMS["nuts_td5"]
    assert (td8.kernel, td8.max_tree_depth, td5.max_tree_depth) == ("nuts", 8, 5)
    assert replace(td5, max_tree_depth=8) == td8  # only the tree depth differs
    assert REFERENCE_ARMS["sa"].kernel == "sa" and REFERENCE_ARMS["ess"].kernel == "ess"
    for config in REFERENCE_ARMS.values():
        assert config.criteria.rhat == 1.01 and config.criteria.ess == 400.0
        assert config.progress_bar


def test_device_tables_agree_and_fit_the_chains() -> None:
    assert reference_run.DEVICES == DEVICES
    for arm, config in REFERENCE_ARMS.items():
        if config.kernel == "ess":
            assert config.chain_method == "vectorized" and DEVICES[arm] == 1
            assert config.num_chains % 2 == 0 and config.num_chains >= 2 * N_PARAMS
        else:
            assert config.chain_method == "parallel"
            assert DEVICES[arm] == config.num_chains <= config.cpus


def test_smoke_settings_are_small() -> None:
    for arm in REFERENCE_ARMS:
        smoke = reference_config(arm, smoke=True)
        assert smoke.design <= 16 and sum(smoke.warmup_rounds) <= 20
        assert smoke.criteria.max_samples <= 40 and not smoke.tight_metric


class ToyModel:
    """Enough of ``BayesianModel`` for ``make_mcmc``: a correlated 2-d normal."""

    def numpyro_model(self) -> Any:
        def model() -> None:
            x = numpyro.sample("x", dist.Normal(0.0, 1.0))
            numpyro.sample("y", dist.Normal(0.9 * x, 0.5))

        return model


class ToyShear:
    """A coordinate change for ``make_mcmc(shear=)``: NUTS then runs on a potential function."""

    def potential(self, bm: Any) -> Any:
        def potential(w: dict[str, Any]) -> Any:
            x, y = w["x"], w["y"]
            return 0.5 * x**2 + 0.5 * ((y - 0.9 * x) / 0.5) ** 2

        return potential


def toy_config(kernel: str) -> PipelineConfig:
    walkers = 8 if kernel == "ess" else 2
    return PipelineConfig(
        kernel=kernel,
        num_chains=walkers,
        # numpyro's SA fails to vmap its init with vectorized chains; the arm runs parallel.
        chain_method="vectorized" if kernel == "ess" else "sequential",
        chunk=20,
        warmup_rounds=(40,) if kernel != "nuts" else (20, 20),
        progress_bar=False,
        criteria=StopCriteria(ess=1e9, max_samples=40, max_divergence_frac=None),
    )


def run_once(kernel: str, folder: Path, criteria: StopCriteria) -> tuple[Checkpoint, StagedWarmup]:
    bm = ToyModel()
    shear = ToyShear() if kernel == "nuts_potential" else None
    kernel = "nuts" if kernel == "nuts_potential" else kernel
    config = replace(toy_config(kernel), criteria=criteria)

    def build(n: int, metric: Any, step: float) -> Any:
        return make_mcmc(bm, config, n, inverse_mass_matrix=metric, step_size=step, shear=shear)

    staged = StagedWarmup(
        build, folder, rounds=config.warmup_rounds, extra_fields=KERNEL_FIELDS[kernel]
    )
    probe = build(config.warmup_rounds[0], None, 1.0)
    stop = Checkpoint.resume(folder, probe, criteria)
    if stop is None:
        init = {
            "x": np.linspace(-1, 1, config.num_chains),
            "y": np.linspace(1, -1, config.num_chains),
        }
        mcmc = staged.run(init, None, seed=1)
        stop = Checkpoint(mcmc, folder, criteria)
    else:
        mcmc = probe
        staged._load()
    if stop.rows:
        stop.rows[-1]["decision"] = None
    sample_chunks(mcmc, stop, seed=len(stop.rows), extra_fields=KERNEL_FIELDS[kernel])
    return stop, staged


@pytest.mark.parametrize("kernel", ["nuts", "nuts_potential", "sa", "ess"])
def test_every_kernel_checkpoints_and_resumes(kernel: str, tmp_path: Path) -> None:
    first, staged = run_once(
        kernel, tmp_path, StopCriteria(ess=1e9, max_samples=40, max_divergence_frac=None)
    )
    base = "nuts" if kernel == "nuts_potential" else kernel
    assert staged.done and len(staged.rows) == len(toy_config(base).warmup_rounds)
    assert first.progress["draws_per_chain"].tolist() == [20, 40]
    assert first.rows[-1]["decision"] == "max_samples"
    for name in ("idata.nc", "mcmc_state.pkl", "progress.csv", "diagnostics.json", "warmup.pkl"):
        assert (tmp_path / name).exists()
    if base == "nuts":
        assert first.rows[-1]["chunk_treedepth_mean"] > 0
        assert 0 < first.rows[-1]["chunk_accept_mean"] <= 1

    # A new process: the warmup is not redone and draws accumulate from the saved state.
    again, staged_again = run_once(
        kernel, tmp_path, StopCriteria(ess=1e9, max_samples=60, max_divergence_frac=None)
    )
    assert len(staged_again.rows) == len(staged.rows)
    assert again.progress["draws_per_chain"].tolist() == [20, 40, 60]
    np.testing.assert_array_equal(again.samples["x"][:, :40], first.samples["x"])
    assert again.seconds_before > 0


def test_deadline_stops_sampling_and_warmup(tmp_path: Path) -> None:
    bm = ToyModel()
    config = toy_config("nuts")

    def build(n: int, metric: Any, step: float) -> Any:
        return make_mcmc(bm, config, n, inverse_mass_matrix=metric, step_size=step)

    staged = StagedWarmup(build, tmp_path, rounds=(20, 20, 20), check=lambda row: ("never",))
    init = {"x": np.asarray([0.1, -0.1]), "y": np.asarray([0.0, 0.2])}
    with pytest.raises(DeadlineReached):
        staged.run(init, None, deadline=time.time() + 1e-3)
    assert len(staged.rows) == 1  # round 1 ran and was saved; round 2 would overrun

    mcmc = build(20, None, 1.0)
    mcmc.warmup(
        __import__("jax").random.PRNGKey(0),
        init_params={k: jnp.asarray(v) for k, v in init.items()},
    )
    stop = Checkpoint(
        mcmc, tmp_path / "s", StopCriteria(ess=1e9, max_divergence_frac=None), deadline=time.time()
    )
    decision = sample_chunks(mcmc, stop, extra_fields=KERNEL_FIELDS["nuts"])
    assert decision == (False, "deadline") and len(stop.rows) == 1


def test_launcher_submits_each_arm_with_its_resources() -> None:
    script = REPO_ROOT / "scripts" / "cluster" / "submit_reference.sh"
    out = subprocess.run(
        ["bash", str(script)],
        env={"SBATCH": "echo", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert len(out) == 4
    by_arm = {line.split("ARM=")[1].split(",")[0]: line for line in out}
    assert set(by_arm) == set(REFERENCE_ARMS)
    for arm, line in by_arm.items():
        assert f"--cpus-per-task={max(REFERENCE_ARMS[arm].cpus, DEVICES[arm])}" in line
        assert f"--job-name=ref_{arm}" in line
    one = subprocess.run(
        ["bash", str(script), "ess"],
        env={"SBATCH": "echo", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert len(one) == 1 and "ARM=ess" in one[0]
    bad = subprocess.run(
        ["bash", str(script), "bogus"],
        env={"SBATCH": "echo", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0


def test_thinned_warmup_round_reports_diagnostics(tmp_path: Path) -> None:
    """A thinned kernel stores ``n // thinning`` warmup draws; the round summary must use them.

    Regression: the SA arm (20,000 warmup iterations, thinning 10) crashed on MASSIVE slicing
    its 2,000 stored draws per chain from position 10,000, leaving an empty array.
    """
    bm = ToyModel()
    config = replace(toy_config("sa"), warmup_rounds=(200,), thinning=10)

    def build(n: int, metric: Any, step: float) -> Any:
        return make_mcmc(bm, config, n, inverse_mass_matrix=metric, step_size=step)

    staged = StagedWarmup(
        build, tmp_path, rounds=config.warmup_rounds, extra_fields=KERNEL_FIELDS["sa"]
    )
    init = {"x": np.linspace(-1, 1, config.num_chains), "y": np.linspace(1, -1, config.num_chains)}
    staged.run(init, None, seed=3)
    row = staged.rows[-1]
    assert row["num_warmup"] == 200
    assert np.isfinite(row["rhat_max"])
    assert (tmp_path / "warmup.pkl").exists()


def test_launcher_run_name_gives_separate_folders_logs_and_jobs() -> None:
    script = REPO_ROOT / "scripts" / "cluster" / "submit_reference.sh"
    env = {"SBATCH": "echo", "PATH": "/usr/bin:/bin"}
    default = subprocess.run(
        ["bash", str(script), "sa"], env=env, capture_output=True, text=True, check=True
    ).stdout
    assert "OUT=outputs/reference/sa " in default and "--job-name=ref_sa " in default
    named = subprocess.run(
        ["bash", str(script), "sa"],
        env={**env, "RUN": "r2"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "OUT=outputs/reference/r2/sa " in named
    assert "--job-name=ref_r2_sa " in named
    assert "--output=outputs/cluster/logs/ref_r2_sa_%j.out" in named
    bad = subprocess.run(
        ["bash", str(script), "sa"], env={**env, "RUN": "a b"}, capture_output=True, text=True
    )
    assert bad.returncode != 0
