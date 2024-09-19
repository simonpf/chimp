"""
chimp.testing
=============

Implements functionality for testing, i.e. evaluating, trained CHIMP retrievals.
"""
from copy import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve import metrics as mtrcs
from pytorch_retrieve.metrics import ScalarMetric
from rich.progress import Progress
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr

from pytorch_retrieve.inference import to_rec


from chimp.tiling import Tiler
from chimp.data.input import get_input_map, get_input_age
from chimp.data.training_data import (
    SingleStepDataset,
    SequenceDataset
)



def get_max_dims(inputs: Dict[str, torch.Tensor]) -> Tuple[int]:
    """
    Calculate maximum input size.

    Args:
        inputs: A dictionary mapping input names to corresponding input
            tensors.

    Return:
        A tuple holding shape of the largest input.
    """
    max_dim = None
    ref_name = None
    for name, tensor in inputs.items():
        if max_dim is None:
            max_dim = max(tensor.shape[-2:])
            ref_name = name
        else:
            dim = max(tensor.shape[-2:])
            if dim > max_dim:
                max_dim = dim
                ref_name = name

    ref_shape = tuple(inputs[ref_name].shape[-2:])
    return ref_shape


def invert_sequence(inpt: Dict[str, List[torch.Tensor]]):
    """
    Invert a sequence of inputs given as a dictionary containing lists of tensors
    to a list containing dicts of tensors.

    Args:
        inpt: The retrieval input as loaded from a chimp.data.training_data.Sequence
            dataset

    Return:
        A list containing the retrieval inputs for each input step.
    """
    n_steps = len(next(iter(inpt.values())))
    inpt_steps = []
    for ind in range(n_steps):
        inpt_steps.append({
            name: tensors[ind] for name, tensors in inpt.items()
        })
    return inpt_steps


def process_tile(
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        input_maps: Union[torch.Tensor, List[torch.Tensor]],
        age_maps: Union[torch.Tensor, List[torch.Tensor]],
        slcs: Tuple[slice],
        drop_steps: Optional[int],
        metrics: Dict[str, List[ScalarMetric]],
        metrics_conditional: Dict[str, ScalarMetric],
        metrics_step: Dict[str, ScalarMetric],
        metrics_forecast: Dict[str, ScalarMetric],
        metrics_persistence: Dict[str, ScalarMetric],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Evaluate predictions for a single tile.

    This function propagates the given inputs through the models and tracks the retrieval
    results using the given metrics.

    Args:
        model: The torch.nn.Module implementing the retrieval.
        inputs: A dictionary mapping input names to corresponding input
            tensors.
        targets: A dictionary mapping target names to corresponding outputs.
        input_maps: The input map representing the input availability or a list of input maps
            for each input time step.
        age_maps: List of tensors containing the age maps for all retrieval input steps.
        slcs: A tuple of slices to extract the valid domain.
        drop_steps: Optional integer specifying the number of time steps to
            drop.
        metrics: A dictionary mapping target names to corresponding metrics
            to compute.
        metrics_conditional: A nested dictionary mapping target names to
            dicts mapping input names to corresponding metrics objects. These
            metrics will only be computed using samples at which the respective
            input is available.
        metrics_step: A dictionary mapping output names to lists of ScalarMetrics
            to calculate conditional on the retrieval step.
        metrics_forecast: A dictionary mapping output names to lists of ScalarMetrics
            to calculate conditional on the forecast step.
        metrics_persistence: A dicitonary mapping output names to lists of ScalarMetrics
            to calculate for the persistence forecast.
        device: The device on which to perform the testing.
        dtype: The dtype to use for the calculation.
    """
    model = model.to(device=device, dtype=dtype)

    with torch.no_grad():

        inputs = to_rec(inputs, device=device, dtype=dtype)
        targets = to_rec(targets, device=device, dtype=dtype)

        has_ref_data = False
        for name, target_k in targets.items():
            if isinstance(target_k, list):
                for targ in target_k:
                    if not targ.mask.all():
                        has_ref_data = True
            else:
                if not target_k.mask.all():
                    has_ref_data = True
        if not has_ref_data:
            return None

        if "lead_time" in inputs:
            n_fc = inputs["lead_time"].shape[1]
        else:
            n_fc = 0

        if model.inference_config is None:
            seq_len = 8
        else:
            seq_len = model.inference_config.input_loader_args.get(
                    "sequence_length",
                    None
            )

        if seq_len is None:
            inputs = invert_sequence(inputs)
            y_pred = {}
            for inpt in inputs:
                for name, tensor in model(inpt).items():
                    y_pred.setdefault(name, []).append(tensor)
        else:
            y_pred = model(inputs)

        for key, y_preds_k in y_pred.items():

            if isinstance(y_preds_k, torch.Tensor):
                y_preds_k = [y_preds_k]

            metrics_k = metrics[key]
            metrics_k_c = metrics_conditional[key]

            if key not in targets:
                continue

            targets_k = targets[key]
            if isinstance(targets_k, torch.Tensor):
                targets_k = [targets_k]

            if n_fc > 0:
                y_preds_k_r = y_preds_k[:-n_fc]
                targets_k_r = targets_k[:-n_fc]
            else:
                y_preds_k_r = y_preds_k
                targets_k_r = targets_k
            input_maps_r = input_maps
            age_maps_r = age_maps

            if drop_steps is not None:
                left = drop_steps
                right = len(y_preds_k_r) - drop_steps
                y_preds_k_r = y_preds_k_r[left:right]
                targets_k_r = targets_k_r[left:right]
                input_maps_r = input_maps[left:right]
                age_maps_r = age_maps[left:right]

            # Evaluate retrieval.
            for step, (y_pred_k, target_k, input_map) in enumerate(zip(
                    y_preds_k_r,
                    targets_k_r,
                    input_maps_r
            )):
                if target_k.mask.all():
                    continue

                y_pred_k = y_pred_k.__getitem__((...,) + slcs)
                target_k = target_k.__getitem__((...,) + slcs)
                input_map = input_map.__getitem__((...,) + slcs)

                y_pred_k_mean = y_pred_k.expected_value()
                for metric in metrics_k:
                    metric = metric.to(device=device)
                    metric.update(y_pred_k_mean, target_k)

                cond = {"step": step * torch.ones_like(target_k)}
                for metric in metrics_step[key]:
                    metric = metric.to(device=device)
                    metric.update(y_pred_k_mean, target_k, conditional=cond)

                metrics_k_c = metrics_conditional.get(key, {})
                for ind, metrics_cond in enumerate(metrics_k_c.values()):

                    for metric in metrics_cond:
                        metric = metric.to(device=device)
                        if len(age_maps_r) > 1:
                            age_map = age_maps_r[step].__getitem__((..., ind) + slcs)
                            target_k_c = target_k.detach().clone()
                            target_k_c.mask[torch.isnan(age_map)] = True
                            metric.update(y_pred_k_mean, target_k_c, conditional={"age": age_map})
                        else:
                            target_k_c = target_k.detach().clone()
                            target_k_c.mask[~input_map[:, ind]] = True
                            if target_k_c.mask.all():
                                continue
                            metric.update(y_pred_k_mean, target_k_c)

            # Evaluate forecast
            if n_fc > 0:
                for step, (y_pred_k, target_k) in enumerate(zip(
                        y_preds_k[-n_fc:],
                        targets_k[-n_fc:],
                )):
                    if target_k.mask.all():
                        continue

                    y_pred_k = y_pred_k.__getitem__((...,) + slcs)
                    y_pred_k_mean = y_pred_k.expected_value()
                    target_k = target_k.__getitem__((...,) + slcs)

                    cond = {"step": step * torch.ones_like(target_k)}

                    for metric in metrics_forecast[key]:
                        metric = metric.to(device=device)
                        metric.update(y_pred_k_mean, target_k, conditional=cond)

                    for metric in metrics_persistence.get(key, []):
                        metric = metric.to(device=device)
                        y_persist = targets_k[-n_fc - 1].__getitem__((...,) + slcs)
                        metric.update(y_persist, target_k, conditional=cond)


def run_tests(
        model: nn.Module,
        test_dataset: Dataset,
        metrics: Dict[str, List[ScalarMetric]],
        conditional: bool = True,
        tile_size: Optional[int] = None,
        batch_size: int = 32,
        device: str = "cuda",
        dtype: str = "float32",
        drop: Optional[List[str]] = None,
        drop_steps: Optional[int] = None
) -> xr.Dataset:
    """
    Evaluate retrieval module on test set.

    Args:
        model: A trained retrieval model.
        test_dataset: A dataset providing access to the test data.
        metrics: A dictionary mapping target names to corresponding
             metrics to evaluate.
        tile_size: A tile size to use for the evaluation.
        device: The device on which to perform the evaluation.
        dtype: The dtype to use.
        drop: Optional list of inputs that will be set to missing.
        drop_steps: Optional number of retrieval steps that will be
            ignored.

    Return:
        A the xarray.Dataset containing the calculated error metrics.
    """
    if drop is None:
        drop = []

    if conditional:
        sequence_length = 0
        forecast = 0
        if hasattr(test_dataset, "sequence_length"):
            sequence_length = test_dataset.sequence_length
        if hasattr(test_dataset, "forecast"):
            forecast = test_dataset.forecast

        if (sequence_length + forecast) > 0:
            tot_steps = sequence_length
            bins = (-tot_steps + 0.5, tot_steps - 0.5, 2 * tot_steps - 1)
            metrics_conditional = {
                target: {
                    inpt.name: [
                        metric.__class__(conditional={"age": bins}) for metric in metrics_t
                    ]
                    for inpt in test_dataset.input_datasets
                }
                for target, metrics_t in metrics.items()
            }
        else:
            metrics_conditional = {
                target: {
                    inpt.name: [copy(metric) for metric in metrics_t]
                    for inpt in test_dataset.input_datasets
                }
                for target, metrics_t in metrics.items()
            }

    else:
        metrics_conditional = {}

    if isinstance(test_dataset, SequenceDataset):
        if test_dataset.include_input_steps:
            metrics_step = {
                target: [
                    mtrcs.Bias(conditional={"step": test_dataset.sequence_length}),
                    mtrcs.MAE(conditional={"step": test_dataset.sequence_length}),
                    mtrcs.MSE(conditional={"step": test_dataset.sequence_length}),
                    mtrcs.SMAPE(conditional={"step": test_dataset.sequence_length}),
                    mtrcs.CorrelationCoef(conditional={"step": test_dataset.sequence_length}),
                ] for target in metrics
            }
        else:
            metrics_step = {}
        if test_dataset.forecast:
            metrics_forecast = {
                target: [
                    mtrcs.Bias(conditional={"step": test_dataset.forecast}),
                    mtrcs.MAE(conditional={"step": test_dataset.sequence_length}),
                    mtrcs.MSE(conditional={"step": test_dataset.sequence_length}),
                    mtrcs.SMAPE(conditional={"step": test_dataset.sequence_length}),
                    mtrcs.CorrelationCoef(conditional={"step": test_dataset.forecast}),
                ] for target in metrics
            }
            if test_dataset.include_input_steps:
                metrics_persistence = {
                    target: [
                        mtrcs.Bias(conditional={"step": test_dataset.forecast}),
                        mtrcs.MAE(conditional={"step": test_dataset.sequence_length}),
                        mtrcs.MSE(conditional={"step": test_dataset.sequence_length}),
                        mtrcs.SMAPE(conditional={"step": test_dataset.sequence_length}),
                        mtrcs.CorrelationCoef(conditional={"step": test_dataset.forecast}),
                    ] for target in metrics
                }
            else:
                metrics_persistence = {}
        else:
            metrics_forecast = {}
            metrics_persistence = {}
    else:
        metrics_step = {}
        metrics_forecast = {}
        metrics_persistence = {}


    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    with Progress() as progress:
        task = progress.add_task("Evaluating retrieval model: ", total=len(data_loader))

        for ind, (inpt, targets) in enumerate(data_loader):

            if tile_size is None:
                tile_size = get_max_dims(inpt)

            lead_time = inpt.pop("lead_time", None)
            tiler = Tiler((inpt, targets), tile_size=tile_size, overlap=tile_size // 4)

            for row_ind in range(tiler.M):
                for col_ind in range(tiler.N):

                    x, y = tiler.get_tile(row_ind, col_ind)
                    x.pop("lead_time", None)
                    input_map = get_input_map(x)
                    age_map = get_input_age(x)
                    if lead_time is not None:
                        x["lead_time"] = lead_time

                    for name in drop:
                        if isinstance(x[name], list):
                            x[name] = [torch.nan * x_s for x_s in x[name]]
                        else:
                            x[name] = torch.nan * x[name]

                    slcs = tiler.get_slices(row_ind, col_ind)
                    process_tile(
                        model, x, y, input_map, age_map, slcs, drop_steps, metrics, metrics_conditional,
                        metrics_step=metrics_step, metrics_forecast=metrics_forecast,
                        metrics_persistence=metrics_persistence, device=device, dtype=dtype
                    )
            progress.update(task, advance=1)

    retrieval_results = {}
    for name, metrics in metrics.items():
        for metric in metrics:
            res_name = name + "_" + metric.name.lower()
            retrieval_results[res_name] = metric.compute().cpu().numpy()
        metrics_c = metrics_conditional.get(name, {})
        for input_name, metrics in metrics_c.items():
            for metric in metrics:
                res_name = name + "_" + metric.name.lower() + "_" + input_name
                retrieval_results[res_name] = (("age",), metric.compute().cpu().numpy())
                age_bins = metric.bins[0]
            retrieval_results["age"] = 0.5 * (age_bins[1:] + age_bins[:-1])
    for name, metrics in metrics_step.items():
        for metric in metrics:
            res_name = name + "_" + metric.name.lower() + "_step"
            retrieval_results[res_name] = (("step",), metric.compute().cpu().numpy())

    if len(retrieval_results) > 0:
        retrieval_results = xr.Dataset(retrieval_results)
    else:
        retrieval_results = None

    forecast_results = {}
    for name, metrics in metrics_forecast.items():
        for metric in metrics:
            res_name = name + "_" + metric.name.lower()
            forecast_results[res_name] = (("step",), metric.compute().cpu().numpy())
    for name, metrics in metrics_persistence.items():
        for metric in metrics:
            res_name = name + "_" + metric.name.lower() + "_persistence"
            forecast_results[res_name] = (("step",), metric.compute().cpu().numpy())

    if len(forecast_results) > 0:
        forecast_results = xr.Dataset(forecast_results)
    else:
        forecast_results = None

    return retrieval_results, forecast_results



@click.argument("model")
@click.argument("test_data_path")
@click.argument("output_filename")
@click.option("-i", "--inputs", "input_datasets", required=True)
@click.option("--reference_datasets")
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=str, default="bfloat16")
@click.option("--tile_size", type=int, default=128)
@click.option("--batch_size", type=int, default=32)
@click.option("--sequence_length", type=int, default=None)
@click.option("--forecast", type=int, default=0)
@click.option("-v", "--verbose", count=True)
@click.option("--drop", type=str, default=None)
@click.option("--drop_steps", type=int, default=None)
def cli(
        model: Path,
        test_data_path: str,
        output_filename: str,
        input_datasets: str,
        reference_datasets: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        tile_size: int = 128,
        verbose: int = 0,
        batch_size: int = 32,
        sequence_length: Optional[int] = 1,
        forecast: int = 0,
        drop: Optional[str] = None,
        drop_steps: Optional[int] = None
) -> int:
    """
    Evaluate model on test data located in TEST_DATA_PATH and write results to
    OUTPUT_FILENAME.nc.
    """
    model = load_model(model).eval()

    input_datasets = [name.strip() for name in  input_datasets.split(",")]
    if len(input_datasets) < 1:
        LOGGER.error(
            "Must specify at least one input dataset using the '--input_datasets' option."
        )
        return 1

    reference_datasets = [name.strip() for name in  reference_datasets.split(",")]
    if len(reference_datasets) < 1:
        LOGGER.error(
            "Must specify at least one reference dataset using the '--reference_datasets' option."
        )
        return 1

    if drop_steps is None:
        sample_rate = 1
    else:
        sample_rate = sequence_length // (sequence_length - 2 * drop_steps)

    test_data = SequenceDataset(
        test_data_path,
        input_datasets=input_datasets,
        reference_datasets=reference_datasets,
        scene_size=-1,
        augment=False,
        validation=True,
        sequence_length=sequence_length,
        forecast=forecast,
        include_input_steps=True,
        sample_rate=sample_rate
    )

    metrics = {
        name: [
            mtrcs.Bias(),
            mtrcs.MSE(),
            mtrcs.MAE(),
            mtrcs.SMAPE(),
            mtrcs.CorrelationCoef()
        ] for name in model.to_config_dict()["output"].keys()
    }

    dtype = getattr(torch, dtype)

    if drop is not None:
        drop = drop.split(",")

    retrieval_results, forecast_results = run_tests(
        model,
        test_data,
        metrics=metrics,
        conditional=True,
        tile_size=tile_size,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        drop=drop,
        drop_steps=drop_steps
    )

    if retrieval_results is not None:
        if forecast_results is not None:
            retrieval_results.to_netcdf(output_filename[:-3] + "_retrieval.nc")
        else:
            retrieval_results.to_netcdf(output_filename)
    if forecast_results is not None:
        if retrieval_results is not None:
            forecast_results.to_netcdf(output_filename[:-3] + "_forecast.nc")
        else:
            forecast_results.to_netcdf(output_filename)
