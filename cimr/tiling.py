from math import ceil
import numpy as np
import torch


def get_start_and_clips(n, tile_size, overlap, soft_end: bool = False):
    """Calculate start indices and numbers of clipped pixels for a given
    side length, tile size and overlap.

    Args:
        n: The image size to tile in pixels.
        tile_size: The size of each tile
        overlap: The number of pixels of overlap.
        soft_end: Allow the last tile to go beyond ``n``, see notes for details

    Return:
        A tuple ``(start, clip)`` containing the start indices of each tile
        and the number of pixels to clip between each neighboring tiles.

    Notes:
        ``soft_end`` is intended to use for cylindrical wrapping of the tiles
        along the horizontal dimension, for example, to have a tile that
        covers the antimeridian. The handling of this special tile should be
        done outside this function.
    """
    start = []
    clip = []
    j = 0
    while j + tile_size < n:
        start.append(j)
        if j > 0:
            clip.append(overlap // 2)
        j = j + tile_size - overlap
    if not soft_end:
        start.append(max(n - tile_size, 0))
    else:
        start.append(j)
    if len(start) > 1:
        clip.append((start[-2] + tile_size - start[-1]) // 2)
    start = start
    clip = clip
    return start, clip


def parse_shape(x):
    """
    Parse largest height and width from a network input that may
    be a dict or a list.

    Args:
        x: The neural network input, which may be a single tensor,
            a list of tensors, or a dict containing input tensors.

    Return:
        A tuple ``(height, width)`` containing the shape of the
        largest input.
    """
    width = 0
    height = 0
    if isinstance(x, dict):
        for key, inpt in x.items():
            height_k, width_k = parse_shape(inpt)
            width = max(width, width_k)
            height = max(height, height_k)
    elif isinstance(x, (list, tuple)):
        for inpt in x:
            height_k, width_k = parse_shape(inpt)
            width = max(width, width_k)
            height = max(height, height_k)
    elif x is None:
        return (0, 0)
    else:
        height, width = x.shape[-2:]
    return (height, width)


class Tiler:
    """
    Helper class that performs two-dimensional tiling of retrieval inputs and
    calculates clipping ranges for the reassembly of tiled predictions.

    Attributes:
        M: The number of tiles along the first image dimension (rows).
        N: The number of tiles along the second image dimension (columns).
    """

    def __init__(self, x, tile_size=512, overlap=32, wrap_columns=False):
        """
        Args:
            x: List of input tensors for the retrieval.
            tile_size: The size of a single tile.
            overlap: The overlap between two subsequent tiles.
            wrap_columns: Apply a circular tiling along the horizontal
                dimension, intended to overcome issues at the antimeridian
        """
        self.x = x
        self.wrap_columns = wrap_columns
        m, n = parse_shape(x)
        self.m = m
        self.n = n

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        if len(tile_size) == 1:
            tile_size = tile_size * 2
        self.tile_size = (min(m, tile_size[0]), min(n, tile_size[1]))
        self.overlap = overlap

        min_len = min(self.tile_size[0], self.tile_size[1])
        if overlap > min_len // 2:
            raise ValueError("Overlap must not exceed the half of the tile size.")

        i_start, i_clip = get_start_and_clips(self.m, tile_size[0], overlap)
        self.i_start = i_start
        self.i_clip = i_clip

        # `soft_end` should be True if circular tiling is expected
        j_start, j_clip = get_start_and_clips(
            self.n, tile_size[1], overlap, soft_end=self.wrap_columns
        )
        self.j_start = j_start
        self.j_clip = j_clip

        self.M = len(i_start)
        self.N = len(j_start)

    def _get_tile_rec(self, i, j, x):
        """
        Get tile in the 'i'th row and 'j'th column of the two
        dimensional tiling.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            List containing the tile extracted from the list
            of input tensors.
        """
        if x is None:
            return None

        if isinstance(x, dict):
            return {
                key: self._get_tile_rec(i, j, inpt) for key, inpt in x.items()
            }
        elif isinstance(x, (list, tuple)):
            return [self._get_tile_rec(i, j, inpt) for inpt in x]

        i_start = self.i_start[i]
        i_end = i_start + self.tile_size[0]
        j_start = self.j_start[j]
        j_end = j_start + self.tile_size[1]

        if self.n % x.shape[-1] != 0:
            raise ValueError(
                f"Input with shape '{x.shape[-2:]}' is incompatible with base "
                f" input size of ({self.m}, {self.n})."
            )
        scl = self.n // x.shape[-1]
        slice_i = slice(i_start // scl, i_end // scl)
        slice_j = slice(j_start // scl, j_end // scl)

        return x[..., slice_i, slice_j]

    def get_tile(self, i, j):
        """
        Get tile in the 'i'th row and 'j'th column of the two
        dimensional tiling.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            List containing the tile extracted from the list
            of input tensors.
        """
        return self._get_tile_rec(i, j, self.x)

    def get_slices(self, i, j):
        """
        Return slices for the clipping of the result tensors.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            Tuple of slices that can be used to clip the retrieval
            results to obtain non-overlapping tiles.
        """
        if i == 0:
            i_clip_l = 0
        else:
            i_clip_l = self.i_clip[i - 1]
        if i >= self.M - 1:
            i_clip_r = self.tile_size[0]
        else:
            i_clip_r = self.tile_size[0] - self.i_clip[i]
        slice_i = slice(i_clip_l, i_clip_r)

        if j == 0:
            j_clip_l = 0
        else:
            j_clip_l = self.j_clip[j - 1]
        if j >= self.N - 1:
            j_clip_r = self.tile_size[1]
        else:
            j_clip_r = self.tile_size[1] - self.j_clip[j]
        slice_j = slice(j_clip_l, j_clip_r)

        return (slice_i, slice_j)

    def get_weights(self, i, j):
        """
        Get weights to reassemble results.

        Args:
            i: Row-index of the tile.
            j: Column-index of the tile.

        Return:
            Numpy array containing weights for the corresponding tile.
        """
        m, n = self.tile_size
        w_i = np.ones((m, n))
        if i > 0:
            trans_start = self.i_start[i]
            # Shift start to right if transition overlaps with
            # antepenultimate tile.
            if i > 1:
                trans_end_prev = self.i_start[i - 2] + self.tile_size[0]
                trans_start = max(trans_start, trans_end_prev)
            zeros = trans_start - self.i_start[i]
            trans_end = self.i_start[i - 1] + self.tile_size[0]
            # Limit transition zone to overlap.
            l_trans = min(trans_end - trans_start, self.overlap)
            w_i[:zeros] = 0.0
            w_i[zeros : zeros + l_trans] = np.linspace(0, 1, l_trans)[..., np.newaxis]

        if i < self.M - 1:
            trans_start = self.i_start[i + 1]
            if i > 0:
                trans_end_prev = self.i_start[i - 1] + self.tile_size[0]
                trans_start = max(trans_start, trans_end_prev)
            trans_end = self.i_start[i] + self.tile_size[0]
            l_trans = min(trans_end - trans_start, self.overlap)

            start = trans_start - self.i_start[i]
            w_i[start : start + l_trans] = np.linspace(1, 0, l_trans)[..., np.newaxis]
            w_i[start + l_trans :] = 0.0

        w_j = np.ones((m, n))
        if j > 0:
            trans_start = self.j_start[j]
            # Shift start to right if transition overlaps with
            # antepenultimate tile and wrapping is not desired
            if j > 1 and not self.wrap_columns:
                trans_end_prev = self.j_start[j - 2] + self.tile_size[1]
                trans_start = max(trans_start, trans_end_prev)
            zeros = trans_start - self.j_start[j]
            trans_end = self.j_start[j - 1] + self.tile_size[1]
            # Limit transition zone to overlap.
            l_trans = min(trans_end - trans_start, self.overlap)
            w_j[:, :zeros] = 0.0
            w_j[:, zeros : zeros + l_trans] = np.linspace(0, 1, l_trans)[np.newaxis]
        elif self.wrap_columns:
            trans_end = (self.j_start[-1] + self.tile_size[1]) - self.n
            l_trans = min(trans_end, self.overlap)
            w_j[:, :l_trans] = np.linspace(0, 1, l_trans)[np.newaxis]

        if j < self.N - 1:
            trans_start = self.j_start[j + 1]
            if j > 0:
                trans_end_prev = self.j_start[j - 1] + self.tile_size[1]
                trans_start = max(trans_start, trans_end_prev)
            trans_end = self.j_start[j] + self.tile_size[1]
            l_trans = min(trans_end - trans_start, self.overlap)

            start = trans_start - self.j_start[j]
            w_j[:, start : start + l_trans] = np.linspace(1, 0, l_trans)[np.newaxis]
            w_j[:, start + l_trans :] = 0.0
        elif self.wrap_columns:
            trans_end = self.j_start[-1] + self.tile_size[1]
            l_trans = min(trans_end % self.n, self.overlap)
            start = self.n - self.j_start[-1]
            w_j[:, start : start + l_trans] = np.linspace(1, 0, l_trans)[np.newaxis]
            w_j[:, start + l_trans :] = 0.0

        return w_i * w_j

    def assemble(self, slices):
        """
        Assemble slices back to original shape using linear interpolation in
        overlap regions.

        Args:
            slices: List of lists of slices.

        Return:
            ``numpy.ndarray`` containing the data from the slices reconstructed
            to the original shape.
        """
        slice_0 = slices[0][0]

        shape = slice_0.shape[:-2] + (self.m, self.n)
        results = np.zeros(shape, dtype=slice_0.dtype)

        for i, row in enumerate(slices):
            for j, slc in enumerate(row):
                i_start = self.i_start[i]
                i_end = i_start + self.tile_size[0]
                row_slice = slice(i_start, i_end)
                j_start = self.j_start[j]
                j_end = j_start + self.tile_size[1]
                # modulo self.n in case self.wrap_columns is True
                col_slice = np.arange(j_start, j_end) % self.n

                results[..., row_slice, col_slice] += self.get_weights(i, j) * slc

        return results


    def initialize_results(self, results_t):
        """
        Initialize containers for assembled results from the results from
        the first tile.

        Args:
            results_t: Retrieval results returned from the first tile.

        Return:
            Depending of the structure of 'results_t', a single numpy.ndarray,
            or a (potentially nested) list or dict of numpy.ndarrays.
        """
        if isinstance(results_t, list):
            return [self.initialize_results(res) for res in results_t]
        if isinstance(results_t, tuple):
            return tuple([self.initialize_results(res) for res in results_t])
        if isinstance(results_t, dict):
            return {
                key: self.initialize_results(val)
                for key, val in results_t.items()
            }
        res = results_t

        ds_row = self.tile_size[0] // res.shape[-2]
        ds_col = self.tile_size[1] // res.shape[-1]
        shape = res.shape[:-2] + (self.m // ds_row, self.n // ds_col)
        return np.zeros(shape, dtype=res.dtype)


    def assemble_tile(self, row_index, col_index, results, results_t):
        """
        Assembles results from a single tile into the assembled result
        containers in 'results'.

        Args:
            row_index: The row index identifying the current tile.
            col_index: The column index identifying the current tile.
            results: Container for the assembled results.
            results_t: Results for the current tile.

        """
        if isinstance(results, (list, tuple)):
            assembled = []
            for res, res_t in zip(results, results_t):
                assembled.append(
                    self.assemble_tile(row_index, col_index, res, res_t)
                )
            if isinstance(results, tuple):
                return tuple(assembled)
            return assembled
        if isinstance(results, dict):
            assembled = {}
            for key in results_t.keys():
                res = results[key]
                res_t = results_t[key]
                assembled[key] = self.assemble_tile(
                    row_index,
                    col_index,
                    res,
                    res_t
                )
            return assembled

        ds_row = self.x.shape[-2] // results.shape[-2]
        ds_col = self.x.shape[-1] // results.shape[-1]

        i_start = self.i_start[row_index]
        i_end = i_start + self.tile_size[0]
        row_slice = slice(i_start // ds_row, i_end // ds_row)
        j_start = self.j_start[col_index]
        j_end = j_start + self.tile_size[1]
        # modulo self.n in case self.wrap_columns is True
        col_slice = (
            np.arange(j_start // ds_col, j_end // ds_col) %
            (self.n // ds_col)
        )

        wgts = self.get_weights(row_index, col_index)[..., ::ds_row, ::ds_col]
        results[..., row_slice, col_slice] += wgts * results_t


    def __iter__(self):

        results = None

        for row_ind in range(self.M):
            for col_ind in range(self.N):

                results_t = yield self.get_tile(row_ind, col_ind)
                if results_t is None:
                    raise ValueError(
                        " Tile received results that are 'None'. You need to "
                        "provide send results for each tile into the tiler "
                        "iterator using 'send'."
                    )

                if results is None:
                    results = self.initialize_results(results_t)
                self.assemble_tile(row_ind, col_ind, results, results_t)

        return results

    def predict(self, predict_fun):
        """
        Applies a prediction function to all tiles in the input
        and assembles the results.

        Args:
            predict_fun: A callable that takes the input from a single tile
                and returns the corresponding predicted results.

        Return:
            The tile-wise results from 'predict_fun' assembled to the original
            size.
        """
        tiler = iter(self)
        x_t = next(tiler)
        try:
            while True:
                results_t = predict_fun(x_t)
                x_t = tiler.send(results_t)
        except StopIteration as exc:
            results = exc.value
        return results


    def __repr__(self):
        return f"Tiler(tile_size={self.tile_size}, overlap={self.overlap})"


def calculate_padding(tensor, multiple_of=32):
    """
    Calculate torch padding dimensions required to pad the input tensor
    to a multiple of 32.

    Args:
        tensor: The tensor to pad.
        multiple_of: Integer of which the spatial dimensions of 'tensor'
            should be a multiple of.

    Return
        A tuple ``(p_l_n, p_r_n, p_l_m, p_r_m)`` containing the
        left and right padding  for the second to last dimension
        (``p_l_m, p_r_m``) and for the last dimension (``p_l_n, p_r_n``).
    """
    shape = tensor.shape

    n = shape[-1]
    d_n = ceil(n / multiple_of) * multiple_of - n
    p_l_n = d_n // 2
    p_r_n = d_n - p_l_n

    m = shape[-2]
    d_m = ceil(m / multiple_of) * multiple_of - m
    p_l_m = d_m // 2
    p_r_m = d_m - p_l_m
    return (p_l_n, p_r_n, p_l_m, p_r_m)
