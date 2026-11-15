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

    def test_two_prompt_ids_can_be_compared_from_one_warm_process_log(self):
        sequences = [
            {"start": {"prompt_id": "i2v", "latent_shapes": [[1, 16, 37, 60, 60]]}, "steps": []},
            {"start": {"prompt_id": "i2v", "latent_shapes": [[1, 16, 97, 60, 60]]}, "steps": []},
            {"start": {"prompt_id": "main", "latent_shapes": [[1, 16, 97, 60, 60]]}, "steps": []},
        ]
        first, second = compare_tool.split_sequential_prompts(sequences)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)
        self.assertEqual(compare_tool.select_longest_video(first)["start"]["prompt_id"], "i2v")
        self.assertEqual(compare_tool.select_longest_video(second)["start"]["prompt_id"], "main")

    def test_three_sequential_executions_distinguish_124_1_and_324_frames(self):
        def sampler(pid, node, t, seconds, refs, keyframes):
            return {
                "start": {
                    "prompt_id": pid, "node_id": node,
                    "latent_shapes": [[1, 24, t, 78, 46], [1, 32, 2, 547]],
                    "refs": refs, "keyframes": keyframes,
                    "steps": 4, "cfg": 1, "sampler": "euler", "scheduler": "simple",
                },
                "steps": [{"elapsed_s": seconds / 4}],
                "finish": {"elapsed_s": seconds, "success": True},
            }
        runs = [
            sampler("i2v124", "10", 37, 40, [{"kind": "image"}], []),
            sampler("i2v124", "92", 97, 261, [], [{"frame": 0}]),
            sampler("main324", "10", 97, 491, [{"kind": "image"}], []),
            sampler("i2v1", "10", 1, 27, [{"kind": "image"}], []),
            sampler("i2v1", "92", 97, 261, [], [{"frame": 0}]),
        ]
        first, second, third = compare_tool.split_sequential_prompts(runs, count=3)
        self.assertEqual([len(first), len(second), len(third)], [2, 1, 2])
        big1 = compare_tool.select_longest_video(first)
        big3 = compare_tool.select_longest_video(third)
        self.assertEqual(big1["start"]["latent_shapes"], big3["start"]["latent_shapes"])
        self.assertEqual(compare_tool.describe(big1)["sampler_seconds"],
                         compare_tool.describe(big3)["sampler_seconds"])
        self.assertEqual(compare_tool.select_longest_video(second)["finish"]["elapsed_s"], 491)
        self.assertEqual(first[0]["start"]["latent_shapes"][0][2], 37)
        self.assertEqual(third[0]["start"]["latent_shapes"][0][2], 1)

    def test_actual_h3_latent_generator_special_cases_single_frame(self):
        import ast
        source = (Path(__file__).resolve().parents[1] /
                  "src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon/SwarmModels.py")
        parsed = ast.parse(source.read_text(encoding="utf-8"))
        wanted = {"temporal_shape", "video_latent_t", "align_frame_count"}
        functions = [node for node in parsed.body
                     if isinstance(node, ast.FunctionDef) and node.name in wanted]
        self.assertEqual({node.name for node in functions}, wanted)
        env = {"H3_FPS": 24, "H3_AUDIO_LATENT_FPS": 40}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), env)
        self.assertEqual(env["temporal_shape"](1), (1, 1, 2))
        self.assertEqual(env["temporal_shape"](124), (124, 37, 207))
        self.assertEqual(env["temporal_shape"](324), (328, 97, 547))

    def test_empty_log_is_not_mistaken_for_fast_sampler(self):
        with self.assertRaisesRegex(ValueError, "SWARM_H3_TRACE"):
            compare_tool.select_longest_video([])


if __name__ == "__main__":
    unittest.main()
