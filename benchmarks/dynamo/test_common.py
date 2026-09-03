from unittest import mock

from benchmarks.dynamo import common

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TinyRunner(common.BenchmarkRunner):
    def __init__(self, skip_fp64, postprocess):
        super().__init__()
        argv = [
            "--accuracy",
            "--training",
            "--float32",
            "--only",
            "probe",
            "--backend",
            "eager",
        ]
        if skip_fp64:
            argv.append("--skip-fp64-check")
        self.args = common.parse_args(argv)
        self.args.iterations = 1
        self.args.amp = postprocess
        self.suite_name = "probe"
        self.model_iter_fn = self.train_step
        self.processed_outputs = 0

    def pick_grad(self, name, training):
        return torch.enable_grad()

    def get_tolerance_and_cosine_flag(self, training, device, name):
        return 1e-4, self.args.cosine

    @property
    def get_output_amp_train_process_func(self):
        return {"probe": self.process_outputs}

    def process_outputs(self, outputs):
        if outputs is None:
            raise AssertionError("Missing reference passed to output processing")
        self.processed_outputs += 1
        return outputs

    def train_step(self, model, inputs, collect_outputs=True):
        model.zero_grad(set_to_none=True)
        out = model(*inputs)
        loss = out.square().mean()
        loss.backward()
        return (
            out.detach().clone(),
            loss.detach().clone(),
            [p.grad.clone() for p in model.parameters()],
        )


class TestBenchmarkAccuracy(TestCase):
    @parametrize(
        "skip_fp64,corrupt,postprocess",
        [
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, False, True),
        ],
    )
    def test_skip_fp64_reference(self, skip_fp64, corrupt, postprocess):
        torch.manual_seed(0)
        runner = TinyRunner(skip_fp64, postprocess)
        model = torch.nn.Linear(4, 4)
        inputs = (torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10,)

        def optimize(fn):
            compiled = torch.compile(fn, backend="eager")

            def invoke(*args, **kwargs):
                result = compiled(*args, **kwargs)
                if corrupt:
                    return result[0] + 1, result[1], result[2]
                return result

            return invoke

        with (
            mock.patch.multiple(
                common, current_name="probe", current_device="cpu", current_batch_size=2
            ),
            mock.patch.object(
                common, "cast_to_fp64", wraps=common.cast_to_fp64
            ) as fp64,
            mock.patch.object(common, "same", wraps=common.same) as compare,
            mock.patch.object(common, "output_signpost", return_value=0),
            mock.patch.object(common, "write_outputs"),
            mock.patch.object(common, "empty_gpu_cache"),
        ):
            status = runner.check_accuracy("probe", model, inputs, optimize, None, None)

        self.assertEqual(status, "fail_accuracy" if corrupt else "pass")
        self.assertEqual(fp64.call_count, 0 if skip_fp64 else 1)
        self.assertGreaterEqual(compare.call_count, 2)
        self.assertFalse(runner.args.cosine)
        if postprocess:
            self.assertEqual(runner.processed_outputs, 2)


instantiate_parametrized_tests(TestBenchmarkAccuracy)

if __name__ == "__main__":
    run_tests()
