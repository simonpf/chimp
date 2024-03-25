"""
chimp.testing
=============

Implements functionality for testing, i.e. evaluating, trained CHIMP retrievals.
"""
from copy import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve import metrics as mtrcs
from pytorch_retrieve.metrics import ScalarMetric
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr


from chimp.tiling import Tiler
from chimp.data.input import get_input_map
from chimp.data.training_data import SingleStepDataset



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
        input_map: torch.Tensor,
        slcs: Tuple[slice],
        metrics: Dict[str, List[ScalarMetric]],
        metrics_conditional: Dict[str, ScalarMetric],
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
        input_map: The input map specifying the input availability.
        slcs: A tuple of slices to extract the valid domain.
        metrics: A dictionary mapping target names to corresponding metrics
            to compute.
        metrics_conditional: A nested dictionary mapping target names to
            dicts mapping input names to corresponding metrics objects. These
            metrics will only be computed using samples at which the respective
            input is available.
        device: The device on which to perform the testing.
        dtype: The dtype to use for the calculation.
    """
    model = model.to(device=device, dtype=dtype)

    with torch.no_grad():

        inputs = {
            name: tensor.to(device=device, dtype=dtype)
            for name, tensor in inputs.items()
        }
        targets = {
            name: tensor.to(device=device, dtype=dtype)
            for name, tensor in targets.items()
        }
        y_pred = model(inputs)

        input_map = input_map.__getitem__((...,) + slcs)
        for key, y_pred_k in y_pred.items():
            metrics_k = metrics[key]
            metrics_k_c = metrics_conditional[key]

            target_k = targets[key]
            if target_k.mask.all():
                continue

            y_pred_k = y_pred_k.__getitem__((...,) + slcs)
            target_k = target_k.__getitem__((...,) + slcs)

            y_pred_k_mean = y_pred_k.expected_value()
            for metric in metrics_k:
                metric = metric.to(device=device)
                metric.update(y_pred_k_mean, target_k)

            metrics_k_c = metrics_conditional.get(key, {})
            for ind, metrics_cond in enumerate(metrics_k_c.values()):
                target_k_c = target_k.detach().clone()
                target_k_c.mask[..., input_map[0, ind]] = True

                if target_k_c.mask.all():
                    continue

                for metric in metrics_cond:
                    metric = metric.to(device=device)
                    metric.update(y_pred_k_mean, target_k_c)


def run_tests(
        model: nn.Module,
        test_dataset: Dataset,
        metrics: Dict[str, List[ScalarMetric]],
        conditional: bool = True,
        tile_size: Optional[int] = None,
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

    data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for ind, (inpt, targets) in enumerate(data_loader):

        print(f"Processing {ind:03}/{len(data_loader)}")
        if ind > 5:
            break

        if tile_size is None:
            tile_size = get_max_dims(inpt)
        tiler = Tiler((inpt, targets), tile_size=tile_size, overlap=0)

        for row_ind in range(tiler.M):
            for col_ind in range(tiler.N):
                print(row_ind, col_ind)

                x, y = tiler.get_tile(row_ind, col_ind)
                input_map = get_input_map(x)
                slcs = tiler.get_slices(row_ind, col_ind)
                process_tile(
                    model, x, y, input_map, slcs, metrics, metrics_conditional,
                    device=device, dtype=dtype
                )

    results = {}
    for name, metrics in metrics.items():
        for metric in metrics:
            res_name = name + "_" + metric.name.lower()
            results[res_name] = metric.compute().cpu().numpy()
        metrics_c = metrics_conditional.get(name, {})
        for input_name, metrics in metrics_c.items():
            for metric in metrics:
                res_name = name + "_" + metric.name.lower() + "_" + input_name
                results[res_name] = metric.compute().cpu().numpy()

    return xr.Dataset(results)



@click.argument("model")
@click.argument("test_data_path")
@click.argument("output_filename")
@click.option("--input_datasets")
@click.option("--reference_datasets")
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=str, default="bfloat16")
@click.option("--tile_size", type=int, default=128)
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
        verbose: int = 0
) -> int:
    """
    Process input files.
    """
    model = load_model(model).eval()

    input_datasets = [name.strip() for name in  input_datasets.split(",")]
    reference_datasets = [name.strip() for name in  reference_datasets.split(",")]

    test_data = SingleStepDataset(
        test_data_path,
        input_datasets=input_datasets,
        reference_datasets=reference_datasets,
        scene_size=-1,
        augment=False,
        validation=True
    )

    metrics = {
        "surface_precip": [
            mtrcs.Bias(),
            mtrcs.MSE(),
            mtrcs.CorrelationCoef()
        ]
    }

    dtype = getattr(torch, dtype)

    results = run_tests(
        model,
        test_data,
        metrics=metrics,
        conditional=True,
        tile_size=tile_size,
        device=device,
        dtype=dtype
    )

    results.to_netcdf(output_filename)
