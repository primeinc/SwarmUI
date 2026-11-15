"""Deterministic tests for selecting and comparing actual H3 sampler traces."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parent / "h3_perf_compare.py"
SPEC = importlib.util.spec_from_file_location("h3_perf_compare", PATH)
compare_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_tool)


def line(event, node, **values):
    data = {"event": event, "prompt_id": "prompt-A", "node_id": node}
    data.update(values)
    return "23:00 [Debug] [ComfyUI-0/STDOUT] [H3PERF] " + json.dumps(data) + "\n"


class H3PerfCompareTests(unittest.TestCase):
    def test_select_long_video_in_two_sampler_run_and_show_real_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            two = Path(folder) / "two.log"
            one = Path(folder) / "one.log"
            two.write_text(
                line("start", "10", latent_shapes=[[1, 16, 37, 60, 60], [1, 4, 2, 210]],
                     refs=[{"kind": "image"}], keyframes=[], steps=4, cfg=1, sampler="res_multistep",
                     scheduler="simple")
                + line("step", "10", step=1, elapsed_s=20)
                + line("finish", "10", elapsed_s=70, success=True)
                + line("start", "92", latent_shapes=[[1, 16, 97, 60, 60], [1, 4, 2, 530]],
                       refs=[], keyframes=[{"frame": 0}], steps=4, cfg=1, sampler="res_multistep",
                       scheduler="simple", text_tokens=[384])
                + line("step", "92", step=1, elapsed_s=65)
                + line("finish", "92", elapsed_s=260, success=True),
                encoding="utf-8")
            one.write_text(
                line("start", "10", latent_shapes=[[1, 16, 97, 60, 60], [1, 4, 2, 530]],
                     refs=[{"kind": "image"}], keyframes=[], steps=4, cfg=1,
                     sampler="res_multistep", scheduler="simple", text_tokens=[384])
                + line("step", "10", step=1, elapsed_s=123)
                + line("finish", "10", elapsed_s=492, success=True),
                encoding="utf-8")
            pair = compare_tool.compare(
                compare_tool.select_longest_video(compare_tool.parse_trace(two)),
                compare_tool.select_longest_video(compare_tool.parse_trace(one)))
            self.assertEqual(pair["image_to_video"]["sampler_node"], "92")
            self.assertEqual(pair["main_video"]["sampler_node"], "10")
            self.assertEqual(pair["image_to_video"]["cfg"], pair["main_video"]["cfg"])
            self.assertIn("ref_blocks", pair["different"])
            self.assertIn("keyframes", pair["different"])
            self.assertEqual(pair["main_video"]["average_step_seconds"], 123.0)
            self.assertEqual(pair["image_to_video"]["average_step_seconds"], 65.0)

    def test_empty_log_is_not_mistaken_for_fast_sampler(self):
        with self.assertRaisesRegex(ValueError, "SWARM_H3_TRACE"):
            compare_tool.select_longest_video([])


if __name__ == "__main__":
    unittest.main()
