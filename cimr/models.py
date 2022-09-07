"""
cimr.models
===========

Machine learning models for CIMR.
"""
import torch
from torch import nn

from cimr.utils import MISSING, MASK


class SymmetricPadding(nn.Module):
    """
    Network module implementing symmetric padding.

    This is just a wrapper around torch's ``nn.functional.pad`` with mode
    set to 'replicate'.
    """

    def __init__(self, amount):
        super().__init__()
        if isinstance(amount, int):
            self.amount = [amount] * 4
        else:
            self.amount = amount

    def forward(self, x):
        return nn.functional.pad(x, self.amount, "replicate")


class SeparableConv(nn.Sequential):
    """
    Depth-wise separable convolution using with kernel size 3x3.
    """

    def __init__(self, channels_in, channels_out, size=7):
        super().__init__(
            nn.Conv2d(
                channels_in,
                channels_in,
                kernel_size=7,
                groups=channels_in
            ),
            #nn.BatchNorm2d(channels_in),
            nn.GroupNorm(channels_in, channels_in),
            nn.Conv2d(channels_in, channels_out, kernel_size=1),
        )

class ConvNextBlock(nn.Module):
    def __init__(self, n_channels, n_channels_out=None, size=7, activation=nn.GELU):
        super().__init__()

        if n_channels_out is None:
            n_channels_out = n_channels
        self.body = nn.Sequential(
            SymmetricPadding(3),
            SeparableConv(n_channels, 2 * n_channels_out, size=size),
            activation(),
            nn.Conv2d(2 * n_channels_out, n_channels_out, kernel_size=1),
        )

        if n_channels != n_channels_out:
            self.projection = nn.Conv2d(n_channels, n_channels_out, kernel_size=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        y = self.body(x)
        return y + self.projection(x)


class DownsamplingBlock(nn.Sequential):
    """
    Xception downsampling block.
    """

    def __init__(self, channels_in, channels_out, bn_first=True):
        if bn_first:
            blocks = [
                #nn.BatchNorm2d(channels_in),
                nn.GroupNorm(channels_in, channels_in),
                nn.Conv2d(channels_in, channels_out, kernel_size=2, stride=2)
            ]
        else:
            blocks = [
                nn.Conv2d(channels_in, channels_out, kernel_size=2, stride=2),
                nn.GroupNorm(channels_out, channels_out),
                #nn.BatchNorm2d(channels_out)
            ]
        super().__init__(*blocks)


class DownsamplingStage(nn.Sequential):
    def __init__(self, channels_in, channels_out, n_blocks, size=7):
        blocks = [DownsamplingBlock(channels_in, channels_out)]
        for i in range(n_blocks):
            blocks.append(ConvNextBlock(channels_out, size=size))
        super().__init__(*blocks)


class UpsamplingStage(nn.Module):
    """
    Xception upsampling block.
    """
    def __init__(self, channels_in, channels_skip, channels_out, size=7):
        """
        Args:
            n_channels: The number of incoming and outgoing channels.
        """
        super().__init__()
        self.upsample = nn.Upsample(mode="bilinear",
                                    scale_factor=2,
                                    align_corners=False)
        self.block = nn.Sequential(
            nn.Conv2d(channels_in + channels_skip, channels_out, kernel_size=1),
            ConvNextBlock(channels_out, size=size)
        )

    def forward(self, x, x_skip):
        """
        Propagate input through block.
        """
        x_up = self.upsample(x)
        if x_skip is not None:
            x_merged = torch.cat([x_up, x_skip], 1)
        else:
            x_merged = x_up
        return self.block(x_merged)


class CIMRModel(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        self.head_visir = nn.Sequential(
            nn.Conv2d(5, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            DownsamplingBlock(16, 16),
            ConvNextBlock(16),
        )

        self.head_geo = nn.Sequential(
            nn.Conv2d(11, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
        )

        #
        # MW heads
        #
        self.head_mw_90 = nn.Sequential(
            nn.Conv2d(2, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.head_mw_160 = nn.Sequential(
            nn.Conv2d(2, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.head_mw_183 = nn.Sequential(
            nn.Conv2d(5, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.merge_mw = nn.Sequential(
            nn.Conv2d(48 + 96, 2 * 96, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 96, 96, kernel_size=1),
        )

        #
        # Downsampling stages
        #

        self.down_stage_4 = nn.Sequential(
            nn.Conv2d(32, 2 * 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 32, 48, kernel_size=1),
            DownsamplingBlock(48, 2 * 48),
            ConvNextBlock(2 * 48),
            ConvNextBlock(2 * 48),
        )

        self.down_stage_8 = nn.Sequential(
            DownsamplingBlock(96, 2 * 96),
            ConvNextBlock(2 * 96),
            ConvNextBlock(2 * 96),
            ConvNextBlock(2 * 96),
        )

        self.down_stage_16 = nn.Sequential(
            DownsamplingBlock(2 * 96, 4 * 96),
            ConvNextBlock(4 * 96),
            ConvNextBlock(4 * 96),
            ConvNextBlock(4 * 96),
        )

        #self.down_stage_32 = nn.Sequential(
        #    DownsamplingBlock(4 * 96, 8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #)

        #self.up_stage_32 = UpsamplingStage(
        #    8 * 96,
        #    4 * 96,
        #    4 * 96
        #)

        self.up_stage_16 = UpsamplingStage(
            4 * 96,
            2 * 96,
            2 * 96
        )

        self.up_stage_8 = UpsamplingStage(
            2 * 96,
            96,
            1 * 96
        )

        self.up_stage_4 = UpsamplingStage(
            96,
            32,
            1 * 96
        )

        self.up_stage_2 = UpsamplingStage(
            96,
            5,
            96
        )

        self.head = nn.Sequential(
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, n_outputs, 1),
        )


    def forward(self, x):

        in_geo = x["geo"]        # n_chans = 11
        in_visir = x["visir"]    # n_chans = 5
        in_mw_90 = x["mw_90"]    # n_chans = 2
        in_mw_160 = x["mw_160"]  # n_chans = 2
        in_mw_183 = x["mw_183"]  # n_chans = 5

        x_visir = self.head_visir(in_visir)
        x_geo = self.head_geo(in_geo)
        x_mw_90 = self.head_mw_90(in_mw_90)
        x_mw_160 = self.head_mw_160(in_mw_160)
        x_mw_183 = self.head_mw_183(in_mw_183)

        x_4 = torch.cat([x_visir, x_geo], 1)       # n_chans = 32
        x_8 = self.down_stage_4(x_4)   # n_chans = 96

        x_8 = self.merge_mw(torch.cat([x_8, x_mw_90, x_mw_160, x_mw_183], 1))

        x_16 = self.down_stage_8(x_8)
        x_32 = self.down_stage_16(x_16)
        #x_64 = self.down_stage_32(x_32)
        #x_32_u = self.up_stage_32(x_64, x_32)
        x_16_u = self.up_stage_16(x_32, x_16)
        x_8_u = self.up_stage_8(x_16_u, x_8)
        x_4_u = self.up_stage_4(x_8_u, x_4)
        x_2_u = self.up_stage_2(x_4_u, in_visir)

        return self.head(x_2_u)


class CIMRSmol(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        self.head_visir = nn.Sequential(
            nn.Conv2d(5, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            DownsamplingBlock(4, 4),
            ConvNextBlock(4),
        )

        self.head_geo = nn.Sequential(
            nn.Conv2d(11, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )

        #
        # MW heads
        #
        self.head_mw_90 = nn.Sequential(
            nn.Conv2d(2, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )
        self.head_mw_160 = nn.Sequential(
            nn.Conv2d(2, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
            ConvNextBlock(4),
        )
        self.head_mw_183 = nn.Sequential(
            nn.Conv2d(5, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
            ConvNextBlock(4),
        )
        self.merge_mw = nn.Sequential(
            nn.Conv2d(12 + 24, 2 * 24, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 24, 24, kernel_size=1),
        )

        #
        # Downsampling stages
        #

        self.down_stage_4 = nn.Sequential(
            nn.Conv2d(8, 2 * 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 8, 12, kernel_size=1),
            DownsamplingBlock(12, 2 * 12),
            ConvNextBlock(2 * 12),
        )

        self.down_stage_8 = nn.Sequential(
            DownsamplingBlock(24, 2 * 24),
            ConvNextBlock(2 * 24),
        )

        self.down_stage_16 = nn.Sequential(
            DownsamplingBlock(2 * 24, 4 * 24),
            ConvNextBlock(4 * 24),
        )

        self.up_stage_16 = UpsamplingStage(
            4 * 24,
            2 * 24,
            2 * 24
        )

        self.up_stage_8 = UpsamplingStage(
            2 * 24,
            24,
            1 * 24
        )

        self.up_stage_4 = UpsamplingStage(
            24,
            8,
            24
        )

        self.up_stage_2 = UpsamplingStage(
            24,
            5,
            24
        )

        self.head = nn.Sequential(
            nn.Conv2d(24, 24, 1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, n_outputs, 1),
        )


    def forward(self, x):

        in_geo = x["geo"]        # n_chans = 11
        in_visir = x["visir"]    # n_chans = 5
        in_mw_90 = x["mw_90"]    # n_chans = 2
        in_mw_160 = x["mw_160"]  # n_chans = 2
        in_mw_183 = x["mw_183"]  # n_chans = 5

        x_visir = self.head_visir(in_visir)
        x_geo = self.head_geo(in_geo)
        x_mw_90 = self.head_mw_90(in_mw_90)
        x_mw_160 = self.head_mw_160(in_mw_160)
        x_mw_183 = self.head_mw_183(in_mw_183)

        x_4 = torch.cat([x_visir, x_geo], 1)       # n_chans = 32
        x_8 = self.down_stage_4(x_4)   # n_chans = 96

        x_8 = self.merge_mw(torch.cat([x_8, x_mw_90, x_mw_160, x_mw_183], 1))

        x_16 = self.down_stage_8(x_8)
        x_32 = self.down_stage_16(x_16)
        #x_64 = self.down_stage_32(x_32)
        #x_32_u = self.up_stage_32(x_64, x_32)
        x_16_u = self.up_stage_16(x_32, x_16)
        x_8_u = self.up_stage_8(x_16_u, x_8)
        x_4_u = self.up_stage_4(x_8_u, x_4)
        x_2_u = self.up_stage_2(x_4_u, in_visir)

        return self.head(x_2_u)



class CIMRSequenceModel(nn.Module):
    def __init__(self, n_outputs, n_hidden=16):

        super().__init__()
        self.n_hidden = n_hidden

        self.head_visir = nn.Sequential(
            nn.Conv2d(n_hidden + 5, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            DownsamplingBlock(16, 16),
            ConvNextBlock(16),
        )

        self.head_geo = nn.Sequential(
            nn.Conv2d(11, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
        )

        #
        # MW heads
        #
        self.head_mw_90 = nn.Sequential(
            nn.Conv2d(2, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.head_mw_160 = nn.Sequential(
            nn.Conv2d(2, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.head_mw_183 = nn.Sequential(
            nn.Conv2d(5, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.merge_mw = nn.Sequential(
            nn.Conv2d(48 + 96, 2 * 96, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 96, 96, kernel_size=1),
        )

        #
        # Downsampling stages
        #

        self.down_stage_4 = nn.Sequential(
            nn.Conv2d(2 * n_hidden + 32, 2 * 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 32, 48, kernel_size=1),
            DownsamplingBlock(48, 2 * 48),
            ConvNextBlock(2 * 48),
            ConvNextBlock(2 * 48),
        )

        self.down_stage_8 = nn.Sequential(
            DownsamplingBlock(4 * n_hidden + 96, 2 * 96),
            ConvNextBlock(2 * 96),
            ConvNextBlock(2 * 96),
            ConvNextBlock(2 * 96),
        )

        self.down_stage_16 = nn.Sequential(
            DownsamplingBlock(8 * n_hidden + 2 * 96, 4 * 96),
            ConvNextBlock(4 * 96),
            ConvNextBlock(4 * 96),
            ConvNextBlock(4 * 96),
        )

        #self.down_stage_32 = nn.Sequential(
        #    DownsamplingBlock(4 * 96, 8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #)

        #self.up_stage_32 = UpsamplingStage(
        #    8 * 96,
        #    4 * 96,
        #    4 * 96
        #)

        self.hidden_32 = nn.Sequential(
            ConvNextBlock(4 * 96 + 16 * n_hidden, n_channels_out=16 * n_hidden),
            ConvNextBlock(16 * n_hidden),
            ConvNextBlock(16 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_16 = UpsamplingStage(
            4 * 96,
            2 * 96,
            2 * 96
        )

        self.hidden_16 = nn.Sequential(
            ConvNextBlock(2 * 96 + 8 * n_hidden, n_channels_out=8 * n_hidden),
            ConvNextBlock(8 * n_hidden),
            ConvNextBlock(8 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_8 = UpsamplingStage(
            2 * 96,
            96,
            1 * 96
        )


        self.hidden_8 = nn.Sequential(
            ConvNextBlock(96 + 4 * n_hidden, n_channels_out=4 * n_hidden),
            ConvNextBlock(4 * n_hidden),
            ConvNextBlock(4 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_4 = UpsamplingStage(
            96,
            32,
            1 * 96
        )

        self.hidden_4 = nn.Sequential(
            ConvNextBlock(96 + 2 * n_hidden, n_channels_out=2*n_hidden),
            ConvNextBlock(2 * n_hidden),
            ConvNextBlock(2 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_2 = UpsamplingStage(
            96,
            5,
            96
        )

        self.hidden_2 = nn.Sequential(
            ConvNextBlock(96 + n_hidden, n_channels_out=n_hidden),
            ConvNextBlock(n_hidden),
            ConvNextBlock(n_hidden, activation=nn.Sigmoid),
        )

        self.head = nn.Sequential(
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, n_outputs, 1),
        )

    def init_hidden(self, x_seq):
        t = x_seq[0]["visir"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]

        hidden = []
        for i in range(6):
            scl = 2 ** i
            t = torch.zeros((b, self.n_hidden * scl, h // scl, w // scl))
            t.to(dtype=dtype, device=device)
            hidden.append(t)
        return hidden

    def forward(self, x_seq):

        hidden_new = self.init_hidden(x_seq)

        y_seq = []

        for x in x_seq:

            hidden_old = hidden_new
            hidden_new = []

            in_geo = x["geo"]        # n_chans = 11
            in_visir = x["visir"]    # n_chans = 5
            in_mw_90 = x["mw_90"]    # n_chans = 2
            in_mw_160 = x["mw_160"]  # n_chans = 2
            in_mw_183 = x["mw_183"]  # n_chans = 5

            x_visir = self.head_visir(torch.cat([in_visir, hidden_old[0]], 1))
            x_geo = self.head_geo(in_geo)
            x_mw_90 = self.head_mw_90(in_mw_90)
            x_mw_160 = self.head_mw_160(in_mw_160)
            x_mw_183 = self.head_mw_183(in_mw_183)

            x_4 = torch.cat([x_visir, x_geo], 1) # n_chans = 32 + 2 * n_hidden
            x_8 = self.down_stage_4(torch.cat([x_4, hidden_old[1]], 1))                       # n_chans = 96 + 4 * n_hidde)

            x_8 = self.merge_mw(torch.cat([x_8, x_mw_90, x_mw_160, x_mw_183], 1))

            x_16 = self.down_stage_8(torch.cat([x_8, hidden_old[2]], 1))
            x_32 = self.down_stage_16(torch.cat([x_16, hidden_old[3]], 1))

            hidden = self.hidden_32(torch.cat([x_32, hidden_old[4]], 1))
            hidden_new.append(hidden)

            #x_64 = self.down_stage_32(x_32)
            #x_32_u = self.up_stage_32(x_64, x_32)
            x_16_u = self.up_stage_16(x_32, x_16)
            hidden = self.hidden_16(torch.cat([x_16_u, hidden_old[3]], 1))
            hidden_new.append(hidden)

            x_8_u = self.up_stage_8(x_16_u, x_8)
            hidden = self.hidden_8(torch.cat([x_8_u, hidden_old[2]], 1))
            hidden_new.append(hidden)

            x_4_u = self.up_stage_4(x_8_u, x_4)
            hidden = self.hidden_4(torch.cat([x_4_u, hidden_old[1]], 1))
            hidden_new.append(hidden)
            x_2_u = self.up_stage_2(x_4_u, in_visir)
            hidden = self.hidden_2(torch.cat([x_2_u, hidden_old[0]], 1))
            hidden_new.append(hidden)

            hidden_new = hidden_new[::-1]
            y_seq.append(self.head(x_2_u))


        return y_seq



class CIMRSmolSequenceModel(nn.Module):
    def __init__(self, n_outputs, n_hidden=2):

        super().__init__()
        self.n_hidden = n_hidden

        self.head_visir = nn.Sequential(
            nn.Conv2d(n_hidden + 5, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            DownsamplingBlock(4, 4),
            ConvNextBlock(4),
        )

        self.head_geo = nn.Sequential(
            nn.Conv2d(11, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )

        #
        # MW heads
        #
        self.head_mw_90 = nn.Sequential(
            nn.Conv2d(2, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )
        self.head_mw_160 = nn.Sequential(
            nn.Conv2d(2, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )
        self.head_mw_183 = nn.Sequential(
            nn.Conv2d(5, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )
        self.merge_mw = nn.Sequential(
            nn.Conv2d(12 + 8, 2 * 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 8, 8, kernel_size=1),
        )

        #
        # Downsampling stages
        #

        self.down_stage_4 = nn.Sequential(
            nn.Conv2d(2 * n_hidden + 8, 2 * 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 8, 8, kernel_size=1),
            DownsamplingBlock(8, 8),
            ConvNextBlock(8),
        )

        #self.down_stage_8 = nn.Sequential(
        #    DownsamplingBlock(4 * n_hidden + 8, 2 * 24),
        #    ConvNextBlock(2 * 24),
        #)

        #self.down_stage_16 = nn.Sequential(
        #    DownsamplingBlock(8 * n_hidden + 2 * 24, 4 * 24),
        #    ConvNextBlock(4 * 24),
        #)

        #self.down_stage_32 = nn.Sequential(
        #    DownsamplingBlock(4 * 24, 8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #)

        #self.up_stage_32 = UpsamplingStage(
        #    8 * 96,
        #    4 * 96,
        #    4 * 96
        #)

        #self.hidden_32 = nn.Sequential(
        #    ConvNextBlock(4 * 24 + 16 * n_hidden, n_channels_out=16 * n_hidden),
        #    ConvNextBlock(16 * n_hidden),
        #    ConvNextBlock(16 * n_hidden, activation=nn.Sigmoid),
        #)

        #self.up_stage_16 = UpsamplingStage(
        #    4 * 24,
        #    2 * 24,
        #    2 * 24
        #)

        #self.hidden_16 = nn.Sequential(
        #    ConvNextBlock(2 * 24 + 8 * n_hidden, n_channels_out=8 * n_hidden),
        #    ConvNextBlock(8 * n_hidden),
        #    ConvNextBlock(8 * n_hidden, activation=nn.Sigmoid),
        #)

        #self.up_stage_8 = UpsamplingStage(
        #    2 * 24,
        #    24,
        #    1 * 24
        #)


        self.hidden_8 = nn.Sequential(
            ConvNextBlock(8 + 4 * n_hidden, n_channels_out=4 * n_hidden),
            ConvNextBlock(4 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_4 = UpsamplingStage(
            8,
            8,
            1 * 8
        )

        self.hidden_4 = nn.Sequential(
            ConvNextBlock(8 + 2 * n_hidden, n_channels_out=2*n_hidden),
            ConvNextBlock(2 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_2 = UpsamplingStage(
            8,
            5,
            8
        )

        self.hidden_2 = nn.Sequential(
            ConvNextBlock(8 + n_hidden, n_channels_out=n_hidden),
            ConvNextBlock(n_hidden, activation=nn.Sigmoid),
        )

        self.head = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, n_outputs, 1),
        )

    def init_hidden(self, x_seq):
        t = x_seq[0]["visir"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]

        hidden = []
        for i in range(3):
            scl = 2 ** i
            t = torch.zeros(
                (b, self.n_hidden * scl, h // scl, w // scl),
                device=device,
                dtype=dtype
            )
            t.to(dtype=dtype, device=device)
            hidden.append(t)
        return hidden

    def forward(self, x_seq):

        hidden_new = self.init_hidden(x_seq)

        y_seq = []

        for x in x_seq:

            hidden_old = hidden_new
            hidden_new = []

            in_geo = x["geo"]        # n_chans = 11
            in_visir = x["visir"]    # n_chans = 5
            in_mw_90 = x["mw_90"]    # n_chans = 2
            in_mw_160 = x["mw_160"]  # n_chans = 2
            in_mw_183 = x["mw_183"]  # n_chans = 5

            x_visir = self.head_visir(torch.cat([in_visir, hidden_old[0]], 1))
            x_geo = self.head_geo(in_geo)
            x_mw_90 = self.head_mw_90(in_mw_90)
            x_mw_160 = self.head_mw_160(in_mw_160)
            x_mw_183 = self.head_mw_183(in_mw_183)

            x_4 = torch.cat([x_visir, x_geo], 1) # n_chans = 32 + 2 * n_hidden
            x_8 = self.down_stage_4(torch.cat([x_4, hidden_old[1]], 1))                       # n_chans = 96 + 4 * n_hidde)

            x_8 = self.merge_mw(torch.cat([x_8, x_mw_90, x_mw_160, x_mw_183], 1))

            #x_16 = self.down_stage_8(torch.cat([x_8, hidden_old[2]], 1))
            #x_32 = self.down_stage_16(torch.cat([x_16, hidden_old[3]], 1))

            #hidden = self.hidden_32(torch.cat([x_32, hidden_old[4]], 1))
            #hidden_new.append(hidden)

            #x_64 = self.down_stage_32(x_32)
            #x_32_u = self.up_stage_32(x_64, x_32)
            #x_16_u = self.up_stage_16(x_32, x_16)
            #hidden = self.hidden_16(torch.cat([x_16, hidden_old[3]], 1))
            #hidden_new.append(hidden)

            #x_8_u = self.up_stage_8(x_16, x_8)
            hidden = self.hidden_8(torch.cat([x_8, hidden_old[2]], 1))
            hidden_new.append(hidden)

            x_4_u = self.up_stage_4(x_8, x_4)
            hidden = self.hidden_4(torch.cat([x_4_u, hidden_old[1]], 1))
            hidden_new.append(hidden)
            x_2_u = self.up_stage_2(x_4_u, in_visir)
            hidden = self.hidden_2(torch.cat([x_2_u, hidden_old[0]], 1))
            hidden_new.append(hidden)

            hidden_new = hidden_new[::-1]
            y_seq.append(self.head(x_2_u))


        return y_seq


class CIMRSeviri(nn.Module):
    """
    The CIMR Seviri baseline model, which only uses SEVIRI observations
    for the retrieval.
    """
    def __init__(
            self,
            n_stages,
            features,
            n_outputs,
            n_blocks=2
    ):
        """
        Args:
            n_stages: The number of stages in the encode
            features: The base number of features.
            n_outputs: The number of outputs of the model.
            n_blocks: The number of blocks in each stage.
        """
        super().__init__()

        n_channels_in = 11

        if not isinstance(n_blocks, list):
            n_blocks = [n_blocks] * n_stages

        stages = []
        ch_in = n_channels_in
        ch_out = features
        for i in range(n_stages):
            stages.append(DownsamplingStage(ch_in, ch_out, n_blocks[i]))
            ch_in = ch_out
            ch_out = ch_out * 2
        self.down_stages = nn.ModuleList(stages)

        stages =[]
        ch_out = ch_in // 2
        for i in range(n_stages):
            ch_skip = ch_out if i < n_stages - 1 else n_channels_in
            stages.append(UpsamplingStage(ch_in, ch_skip, ch_out))
            ch_in = ch_out
            ch_out = ch_out // 2 if i < n_stages - 2 else features
        self.up_stages = nn.ModuleList(stages)

        self.up = UpsamplingStage(features, 0, features)
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, n_outputs, kernel_size=1),
        )


    def forward(self, x):
        """Propagate input though model."""

        skips = []
        y = x["geo"]
        for stage in self.down_stages:
            skips.append(y)
            y = stage(y)

        skips.reverse()
        for skip, stage in zip(skips, self.up_stages):
            y = stage(y, skip)

        return self.head(self.up(y, None))


class CIMRSeqSeviri(nn.Module):
    """
    The CIMR Seviri baseline model, which only uses SEVIRI observations
    for the retrieval.
    """
    def __init__(
            self,
            n_stages,
            features,
            n_outputs,
            n_blocks=2,
            n_hidden=4
    ):
        """
        Args:
            n_stages: The number of stages in the encoder
            features: The base number of features.
            n_outputs: The number of outputs of the model.
            n_blocks: The number of blocks in each stage.
        """
        super().__init__()
        self.n_stages = n_stages
        self.n_features = features

        n_channels_in = 11

        if not isinstance(n_blocks, list):
            n_blocks = [n_blocks] * n_stages

        stages = []
        ch_in = n_channels_in
        ch_out = features
        for i in range(n_stages):
            if i == 0:
                stages.append(DownsamplingStage(ch_in, ch_out, n_blocks[i], size=3))
            else:
                stages.append(DownsamplingStage(2 * ch_in, ch_out, n_blocks[i], size=3))
            ch_in = ch_out
            ch_out = ch_out * 2
        self.down_stages = nn.ModuleList(stages)

        self.center = ConvNextBlock(2 * ch_in, ch_in)

        stages =[]
        ch_out = ch_in // 2
        for i in range(n_stages):
            ch_skip = 2 * ch_out if i < n_stages - 1 else n_channels_in
            stages.append(UpsamplingStage(2 * ch_in, ch_skip, ch_out, size=3))
            ch_in = ch_out
            ch_out = ch_out // 2 if i < n_stages - 2 else features
        self.up_stages = nn.ModuleList(stages)

        self.up = UpsamplingStage(2 * features, 0, features, 3)
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, n_outputs, kernel_size=1),
        )

    def init_hidden(self, x):

        t = x[0]["geo"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]

        hidden_down = []
        for i in range(self.n_stages):
            scl = 2 ** i
            t = torch.zeros(
                (b, self.n_features * scl, h // scl // 2, w // scl // 2),
                device=device,
                dtype=dtype
            )
            t.to(dtype=dtype, device=device)
            hidden_down.append(t)

        hidden_up = []
        for i in range(self.n_stages - 1, -1, -1):
            scl = 2 ** i
            t = torch.zeros(
                (b, self.n_features * scl, h // scl // 2, w // scl // 2),
                device=device,
                dtype=dtype
            )
            t.to(dtype=dtype, device=device)
            hidden_up.append(t)

        t = torch.zeros(
            (b, self.n_features * scl, h, w),
            device=device,
            dtype=dtype
        )
        t.to(dtype=dtype, device=device)
        hidden_up.append(t)

        return hidden_down, hidden_up


    def forward(self, x_seq, h_states=None):
        """Propagate input though model."""

        if not isinstance(x_seq, list):
            x_seq = [x_seq]

        if h_states is None:
            h_down, h_up = self.init_hidden(x_seq)
        else:
            h_down, h_up = h_states

        results = []

        for i, x in enumerate(x_seq):

            skips = []
            y = x["geo"]

            h_down_new = []
            h_up_new = []

            for hidden, stage in zip(h_down, self.down_stages):
                skips.append(y)
                y = stage(y)
                h_down_new.append(y)
                y = torch.cat([y, hidden], 1)

            skips.reverse()

            y = self.center(y)
            h_up_new.append(y)
            y = torch.cat([y, h_up[0]], 1)

            for hidden, skip, up_stage, in zip(h_up[1:], skips, self.up_stages):
                y = up_stage(y, skip)
                h_up_new.append(y)
                if hidden is not None:
                    y = torch.cat([y, hidden], 1)

            h_down = h_down_new
            h_up = h_up_new

            results.append(self.head(self.up(y, None)))

        return results


class RNN(nn.Module):
    """
    Independent-pixel RNN.
    """
    def __init__(self, n_inputs, n_features, n_outputs):
        """
        """
        super().__init__()
        self.n_features = n_features

        self.body = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(n_inputs, n_features, kernel_size=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(n_features + n_features, n_features, kernel_size=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(n_features + n_features, n_features, kernel_size=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(n_features + n_features, n_features, kernel_size=1),
                nn.GELU(),
            )
        )
        self.head = nn.Sequential(
                nn.Conv2d(n_features + n_features, n_features, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(n_features, n_features, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(n_features, n_outputs, kernel_size=1),
        )


    def init_hidden(self, x_seq):
        t = x_seq[0]["geo"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]
        hidden = []

        for i in range(len(self.body)):
            hidden += [torch.zeros(
                (b, self.n_features, h, w),
                device=device,
                dtype=dtype
            )]
        return hidden


    def forward(self, x_seq, h_states=None):
        """Propagate input though model."""

        if not isinstance(x_seq, list):
            x_seq = [x_seq]

        if h_states is None:
            h_states = self.init_hidden(x_seq)

        results = []

        for i, x in enumerate(x_seq):

            y = x["geo"]
            hidden_new = []

            for hidden, stage in zip(h_states, self.body):
                y = stage(y)
                hidden_new.append(y)
                y = torch.cat([y, hidden], 1)

            results.append(self.head(y))
            h_states = hidden_new

        return results
