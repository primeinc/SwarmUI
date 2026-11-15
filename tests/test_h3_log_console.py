"""CPU-only end-to-end regression tests for the interactive H3 log analyzer backend."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h3_log_console as h3


def event(pid, node, kind, **extra):
    data = {'prompt_id': pid, 'node_id': node, 'event': kind}
    data.update(extra)
    # Actual SwarmUI and ComfyUI logs wrap this tag inside other log prefixes.
    return f'00:15:32.001 [Debug] [ComfyUI-0/STDERR] [INFO] [H3PERF] {json.dumps(data)}\n'


def run_data(pid, node, t, seconds, *, refs=(), keyframes=(), started=True):
    result = event(pid, node, 'start', latent_shapes=[[1, 16, t, 60, 78], [1, 4, 2, 547]],
                   refs=list(refs), keyframes=list(keyframes), steps=4, cfg=1,
                   sampler='res_multistep', scheduler='simple', model_type='MiniMaxH3',
                   weights_resident_mb=2011.5, weights_total_mb=13348)
    for i in range(4):
        result += event(pid, node, 'step', step=i+1, total_steps=4, elapsed_s=seconds/4,
                        callback_s=0.01, weights_resident_mb=2000+i*10,
                        weights_offloaded_mb=11348-i*10)
    if started:
        result += event(pid, node, 'finish', elapsed_s=seconds, success=True, weights_resident_mb=2111)
    return result


class H3LogConsoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.logs = [self.base / s for s in ['i2v.log', 'main.log', 'oneframe.log', 'combined.log']]
        self.logs[0].write_text(
            run_data('i2v', '10', 37, 175, refs=[{'kind': 'image'}]) +
            run_data('i2v', '44', 97, 261, keyframes=[{'frame': 0}]), encoding='utf-8')
        self.logs[1].write_text(run_data('main', '10', 97, 491, refs=[{'kind': 'image'}]), encoding='utf-8')
        self.logs[2].write_text(
            run_data('short', '10', 1, 28, refs=[{'kind': 'image'}]) +
            run_data('short', '88', 97, 252, keyframes=[{'frame': 0}]), encoding='utf-8')
        self.logs[3].write_text(''.join(p.read_text(encoding='utf-8') for p in self.logs[:3]), encoding='utf-8')

    def test_scan_reports_both_sampler_stages_without_relabeling_as_two_videos(self):
        report = h3.build_report('scan', [self.logs[0]])
        self.assertTrue(report['trace_found'])
        self.assertEqual(len(report['prompts']), 1)
        self.assertEqual(len(report['prompts'][0]['samplers']), 2)
        self.assertIn('Sampler elapsed', h3.render(report))
        self.assertEqual(report['prompts'][0]['samplers'][1]['sampler_seconds'], 261)

    def test_legacy_scanner_makes_no_claims_about_actual_latents(self):
        old = self.base / 'before-trace.log'
        old.write_text('00:01:00 [INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14955MB Staged.\n'
                       '00:01:00 [INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 13348MB Staged.\n'
                       '00:07:00 [INFO] Prompt executed in 338.45 seconds\n'
                       '00:07:03 Generated an image in 5.64 min (gen)\n', encoding='utf-8')
        report = h3.build_report('scan', [old])
        self.assertFalse(report['trace_found'])
        self.assertEqual(report['legacy']['prompt_times_s'], [338.45])
        self.assertEqual(report['legacy']['generated_output_counts'], [1])
        self.assertEqual(report['legacy']['preparations'][1]['staged_mb'], 13348)
        self.assertIn('No detailed trace', h3.render(report))

    def test_two_runs_one_log_picks_long_sampler(self):
        combined = self.base / 'two-in-one.log'
        combined.write_text(self.logs[0].read_text(encoding='utf-8')+self.logs[1].read_text(encoding='utf-8'), encoding='utf-8')
        report=h3.build_report('two-in-one', [combined])
        self.assertEqual(report['comparison']['image_to_video']['sampler_seconds'], 261)
        self.assertEqual(report['comparison']['main_video']['sampler_seconds'], 491)
        self.assertAlmostEqual(report['comparison']['main_video']['average_step_seconds']/report['comparison']['image_to_video']['average_step_seconds'], 491/261)
        self.assertEqual(report['sampler_counts'], [2,1])

    def test_three_runs_in_one_log_and_three_files_have_same_sampler_results(self):
        one=h3.build_report('three-in-one', [self.logs[3]])
        separate=h3.build_report('three-files', self.logs[:3])
        for field in ['image_to_video','main_video','one_frame_followup']:
            self.assertEqual(one['comparison'][field]['sampler_seconds'], separate['comparison'][field]['sampler_seconds'])
        self.assertEqual(one['sampler_counts'], [2,1,2])
        self.assertEqual(one['comparison']['one_frame_followup']['sampler_seconds'], 252)
        self.assertEqual(one['comparison']['main_video']['weights_resident_mb_per_step'], [2000,2010,2020,2030])

    def test_two_files_writes_machine_and_human_reports_with_spaces_in_names(self):
        out = self.base / 'folder containing spaces' / 'H3-Reports'
        result = h3.build_report('two-files', self.logs[:2])
        summary, saved = h3.save_report(result, out)
        self.assertIn('B / A step-time ratio', summary)
        self.assertTrue(Path(saved['json']).is_file())
        self.assertEqual(json.loads(Path(saved['json']).read_text(encoding='utf-8'))['mode'], 'two-files')
        self.assertTrue(Path(saved['summary']).is_file())
        self.assertNotIn('integrated_multimodal_description:', summary)

    def test_export_excludes_unrelated_log_lines_and_keeps_structured_events(self):
        report=h3.build_report('extract', [self.logs[0]])
        self.assertEqual(len(report['events']), 12)
        _, saved=h3.save_report(report, self.base / 'reports')
        events=Path(saved['events']).read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(events), 12)
        self.assertEqual(json.loads(events[0])['event'], 'start')
        self.assertNotIn('ComfyUI-0/STDERR', events[0])

    def test_vaes_are_reported_separately_from_sampler_and_legacy_timing(self):
        src = self.base / 'with-vaes.log'
        src.write_text(
            self.logs[1].read_text(encoding='utf-8')
            + '00:09:37 [INFO] [H3VAE] ' + json.dumps({
                'prompt_id': 'main', 'node_id': '80', 'stage': 'video',
                'event': 'start', 'method': 'decode_tiled', 'latent_shape': [1, 16, 97, 60, 78]
            }) + '\n'
            + '00:09:52 [INFO] [H3VAE] ' + json.dumps({
                'prompt_id': 'main', 'node_id': '80', 'stage': 'video',
                'event': 'working', 'elapsed_s': 15
            }) + '\n'
            + '00:10:46 [INFO] [H3VAE] ' + json.dumps({
                'prompt_id': 'main', 'node_id': '80', 'stage': 'video',
                'event': 'finish', 'elapsed_s': 69.0, 'status': 'success'
            }) + '\n'
            + '00:10:47 [INFO] [H3VAE] ' + json.dumps({
                'prompt_id': 'main', 'node_id': '81', 'stage': 'audio',
                'event': 'start', 'method': 'decode'
            }) + '\n'
            + '00:10:48 [INFO] [H3VAE] ' + json.dumps({
                'prompt_id': 'main', 'node_id': '81', 'stage': 'audio',
                'event': 'finish', 'elapsed_s': 1.25, 'status': 'success'
            }) + '\n', encoding='utf-8')
        report = h3.build_report('scan', [src])
        self.assertTrue(report['trace_found'])
        self.assertEqual(report['vae_summary']['video_decode_s'], 69)
        self.assertEqual(report['vae_summary']['audio_decode_s'], 1.25)
        self.assertEqual(report['vae_summary']['finish_count'], 2)
        self.assertFalse(report['vae_summary']['incomplete'])
        summary = h3.render(report)
        self.assertIn('VAE DECODE TIMELINE', summary)
        self.assertIn('69.00 s', summary)
        self.assertIn('1.25 s', summary)

    def test_compare_and_trace_export_include_vaes_but_keep_samplers_separate(self):
        first = self.base / 'first-plus-vae.log'
        first.write_text(
            self.logs[0].read_text(encoding='utf-8')
            + '[INFO] [H3VAE] ' + json.dumps({
                'prompt_id': 'i2v', 'node_id': '100', 'stage': 'video',
                'event': 'finish', 'elapsed_s': 34.5, 'status': 'success'
            }) + '\n', encoding='utf-8')
        result = h3.build_report('two-files', [first, self.logs[1]])
        self.assertEqual(result['sampler_counts'], [2, 1])
        self.assertEqual(result['vae_by_run']['image_to_video']['video_decode_s'], 34.5)
        self.assertEqual(result['vae_by_run']['main_video']['events'], 0)
        self.assertIn('POST-SAMPLER VAE DECODING', h3.render(result))
        export = h3.build_report('extract', [first])
        vae = [x for x in export['events'] if x['trace_kind'] == 'H3VAE']
        self.assertEqual(len(vae), 1)
        self.assertEqual(vae[0]['stage'], 'video')
        self.assertEqual(sum(x['trace_kind'] == 'H3PERF' for x in export['events']), 12)

    def test_incomplete_vae_and_bad_vae_json_are_detected(self):
        log = self.base / 'incomplete-vae.log'
        log.write_text('[H3VAE] ' + json.dumps({
            'prompt_id': 'P', 'node_id': '77', 'event': 'start', 'stage': 'video'
        }) + '\n', encoding='utf-8')
        result = h3.build_report('scan', [log])
        self.assertTrue(result['vae_summary']['incomplete'])
        self.assertEqual(result['vae_summary']['video_decode_s'], 0)
        log.write_text('[H3VAE] {"bad": incomplete\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Malformed \\[H3VAE\\]'):
            h3.build_report('scan', [log])

    def test_missing_trace_errors_without_creating_false_comparison(self):
        old = self.base / 'old.log'
        old.write_text('Prompt executed in 384.81 seconds', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'SWARM_H3_TRACE'):
            h3.build_report('two-files', [self.logs[0], old])
        with self.assertRaisesRegex(ValueError, 'No \\[H3PERF\\]'):
            h3.build_report('extract', [old])

    def test_malformed_trace_is_rejected(self):
        bad=self.base/'bad.log'
        bad.write_text('[H3PERF] {"event": incomplete\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'malformed'):
            h3.build_report('extract', [bad])

    def test_wrong_log_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'expected 3'):
            h3.build_report('three-files', self.logs[:2])

    def test_real_cli_creates_persistent_outputs_and_fails_closed(self):
        cli = Path(h3.__file__)
        outdir=self.base/'reports'
        run=subprocess.run([sys.executable,str(cli),'--mode','two-files',
                            '--log',str(self.logs[0]),'--log',str(self.logs[1]),
                            '--outdir',str(outdir)],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('1.88x',run.stdout)
        self.assertEqual(len(list(outdir.glob('*.json'))),1)
        failed=subprocess.run([sys.executable,str(cli),'--mode','three-files',
                               '--log',str(self.logs[0]),'--outdir',str(outdir)],
                               capture_output=True,text=True)
        self.assertNotEqual(failed.returncode,0)
        self.assertEqual(len(list(outdir.glob('*.json'))),1)

    def test_batch_menu_dispatch_is_not_an_inert_display(self):
        batch=(Path(h3.__file__).parents[1] / 'H3_Log_Analyzer.bat').read_text(encoding='utf-8')
        self.assertIn('call :PICKLOG LOG1',batch)
        self.assertIn('call :PICKLOG LOG3',batch)
        self.assertIn('--mode "%MODE%" --log "%LOG1%" --log "%LOG2%"',batch)
        self.assertIn('goto RUN_THREE_FILES',batch)
        self.assertIn('goto RUN',batch)
        self.assertIn('--self-test',batch)
        self.assertIn('System.Windows.Forms.OpenFileDialog',batch)
        self.assertIn('DisableDelayedExpansion',batch)
        segment = batch.split(':RUN_TWO_FILES\n', 1)[1].split('\n:RUN_THREE_FILES\n', 1)[0]
        self.assertIn('--mode "%MODE%" --log "%LOG1%" --log "%LOG2%"', segment)
        self.assertNotIn('--mode "%MODE%" --log "%LOG1%" --log "%LOG2%"',
                         segment.replace('--mode "%MODE%" --log "%LOG1%" --log "%LOG2%"', '', 1))

if __name__=='__main__':
    unittest.main()
