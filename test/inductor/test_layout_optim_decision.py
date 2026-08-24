# Owner(s): ["module: inductor"]
import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx


def conv_backward_graph(groups: int):
    in_channels = 64
    out_channels = 128

    def fn(grad_output, inp, weight):
        return torch.ops.aten.convolution_backward.default(
            grad_output,
            inp,
            weight,
            [out_channels],
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            groups,
            [True, True, True],
        )

    grad_output = torch.randn(2, out_channels, 8, 8)
    inp = torch.randn(2, in_channels, 8, 8)
    weight = torch.randn(out_channels, in_channels // groups, 1, 1)
    return make_fx(fn)(grad_output, inp, weight)


class TestLayoutOptimDecision(TestCase):
    @config.patch({"layout_optimization": True, "force_layout_optimization": False})
    def test_conv_backward_enables_layout_opt(self):
        with torch.backends.mkldnn.flags(enabled=False):
            gm = conv_backward_graph(groups=1)
            self.assertTrue(GraphLowering.decide_layout_opt(gm, is_inference=True))
            self.assertTrue(GraphLowering.decide_layout_opt(gm, is_inference=False))

    @config.patch({"layout_optimization": True, "force_layout_optimization": False})
    def test_conv_backward_grouped_uses_groups_arg(self):
        with torch.backends.mkldnn.flags(enabled=False):
            gm = conv_backward_graph(groups=2)
            self.assertFalse(GraphLowering.decide_layout_opt(gm, is_inference=True))
            self.assertFalse(GraphLowering.decide_layout_opt(gm, is_inference=False))


if __name__ == "__main__":
    run_tests()
