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
from chimp.data.input import get_input_map
from chimp.data.training_data import (
    SingleStepDataset,
    SequenceDataset
)



def get_max_dims(inputs: Dict[str, torch.Tensor]) -> Tuple[int]:
    """
    Calculate maximum input dimensions.

    Args:
        inputs: A dictionary mapping input names to corresponding output
            tensors.

    Return:
        A tuple holding dimensions of the largest input.
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


def process_tile(
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        input_maps: Union[torch.Tensor, List[torch.Tensor]],
        slcs: Tuple[slice],
        metrics: Dict[str, List[ScalarMetric]],
        metrics_conditional: Dict[str, ScalarMetric],
        metrics_step: Dict[str, ScalarMetric],
        metrics_forecast: Dict[str, ScalarMetric],
        metrics_persistence: Dict[str, ScalarMetric],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
) -> None:
    """
    Process a single tile.

    Args:
        model: The torch.nn.Module implementing the retrieval.
        inputs: A dictionary mapping input names to corresponding input
            tensors.
        targets: A dictionary mapping target names to corresponding outputs.
        input_maps: The input map representing the input availability or a list of input maps
            for each input time step.
        slcs: A tuple of slices to extract the valid domain.
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


        y_pred = model(inputs)

        for key, y_preds_k in y_pred.items():

            if isinstance(y_preds_k, torch.Tensor):
                y_preds_k = [y_preds_k]

            metrics_k = metrics[key]
            metrics_k_c = metrics_conditional[key]

            targets_k = targets[key]
            if isinstance(targets_k, torch.Tensor):
                targets_k = [targets_k]

            # Evaluate retrieval.
            for step, (y_pred_k, target_k, input_map) in enumerate(zip(
                    y_preds_k[:-n_fc],
                    targets_k[:-n_fc],
                    input_maps
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
                    target_k_c = target_k.detach().clone()
                    target_k_c.mask[~input_map[:, ind]] = True

                    if target_k_c.mask.all():
                        continue

                    for metric in metrics_cond:
                        metric = metric.to(device=device)
                        metric.update(y_pred_k_mean, target_k_c)

            # Evaluate forecast
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

                for metric in metrics_persistence[key]:
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
        dtype: str = "float32"
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

    Return:
        A the xarray.Dataset containing the calculated error metrics.
    """
    if conditional:
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
                    mtrcs.MSE(conditional={"step": test_dataset.sequence_length}),
                    mtrcs.CorrelationCoef(conditional={"step": test_dataset.sequence_length}),
                ] for target in metrics
            }
        else:
            metrics_step = {}
        if test_dataset.forecast:
            metrics_forecast = {
                target: [
                    mtrcs.Bias(conditional={"step": test_dataset.forecast}),
                    mtrcs.MSE(conditional={"step": test_dataset.forecast}),
                    mtrcs.CorrelationCoef(conditional={"step": test_dataset.forecast}),
                ] for target in metrics
            }
            if test_dataset.include_input_steps:
                metrics_persistence = {
                    target: [
                        mtrcs.Bias(conditional={"step": test_dataset.forecast}),
                        mtrcs.MSE(conditional={"step": test_dataset.forecast}),
                        mtrcs.CorrelationCoef(conditional={"step": test_dataset.forecast}),
                    ] for target in metrics
                }
        else:
            metrics_step = {}
            metrics_forecast = {}
            metrics_persistence = {}


    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with Progress() as progress:
        task = progress.add_task("Evaluating retrieval model: ", total=len(data_loader))

        for ind, (inpt, targets) in enumerate(data_loader):

            if tile_size is None:
                tile_size = get_max_dims(inpt)

            lead_time = inpt.pop("lead_time", None)
            tiler = Tiler((inpt, targets), tile_size=tile_size, overlap=0)

            for row_ind in range(tiler.M):
                for col_ind in range(tiler.N):

                    x, y = tiler.get_tile(row_ind, col_ind)
                    x.pop("lead_time", None)
                    input_map = get_input_map(x)
                    if lead_time is not None:
                        x["lead_time"] = lead_time

                    slcs = tiler.get_slices(row_ind, col_ind)
                    process_tile(
                        model, x, y, input_map, slcs, metrics, metrics_conditional,
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
                retrieval_results[res_name] = metric.compute().cpu().numpy()
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
@click.option("--input_datasets")
@click.option("--reference_datasets")
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=str, default="bfloat16")
@click.option("--tile_size", type=int, default=128)
@click.option("--batch_size", type=int, default=32)
@click.option("--sequence_length", type=int, default=None)
@click.option("--forecast", type=int, default=0)
@click.option("-v", "--verbose", count=True)
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
        sequence_length: Optional[int] = None,
        forecast: int = 0
) -> int:
    """
    Process input files.
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

    if sequence_length is None:
        test_data = SingleStepDataset(
            test_data_path,
            input_datasets=input_datasets,
            reference_datasets=reference_datasets,
            scene_size=-1,
            augment=False,
            validation=True
        )
    else:
        test_data = SequenceDataset(
            test_data_path,
            input_datasets=input_datasets,
            reference_datasets=reference_datasets,
            scene_size=-1,
            augment=False,
            validation=True,
            sequence_length=sequence_length,
            forecast=forecast,
            include_input_steps=True
        )

    metrics = {
        name: [
            mtrcs.Bias(),
            mtrcs.MSE(),
            mtrcs.CorrelationCoef()
        ] for name in model.to_config_dict()["output"].keys()
    }

    dtype = getattr(torch, dtype)

    retrieval_results, forecast_results = run_tests(
        model,
        test_data,
        metrics=metrics,
        conditional=True,
        tile_size=tile_size,
        device=device,
        dtype=dtype,
        batch_size=batch_size
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
