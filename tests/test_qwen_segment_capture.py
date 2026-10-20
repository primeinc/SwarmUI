import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

PATH = Path(__file__).resolve().parents[1] / 'launchtools' / 'qwen_segment_capture.py'
spec = importlib.util.spec_from_file_location('capture', PATH)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def fixture():
    def node(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}
    return {
        'model': node('UNETLoader', unet_name='qwen-2.1.safetensors'),
        'global': node('LoraLoaderModelOnly', model=['model', 0], lora_name='speed.safetensors', strength_model=1),
        'vae': node('VAELoader', vae_name='qwen-vae.safetensors'),
        'text': node('SwarmTextEncodeAdvanced', prompt='base prompt'),
        'base': node('SwarmKSampler', model=['global', 0], positive=['text', 0], steps=6),
        'basepixels': node('VAEDecode', samples=['base', 0], vae=['vae', 0]),
        'crop': node('SwarmImageCrop', image=['basepixels', 0]),
        'cropmask': node('CropMask', mask=['mask', 0]),
        'mask': node('SolidMask', value=1),
        'encoded': node('VAEEncode', pixels=['crop', 0], vae=['vae', 0]),
        'masked': node('SetLatentNoiseMask', samples=['encoded', 0], mask=['cropmask', 0]),
        'scoped': node('LoraLoaderModelOnly', model=['global', 0], lora_name='face.safetensors', strength_model=0.9),
        'segtext': node('SwarmTextEncodeAdvanced', prompt='segment trigger', images=None),
        'reference': node('ReferenceLatent', conditioning=['segtext', 0], latent=['encoded', 0]),
        'segment': node('SwarmKSampler', model=['scoped', 0], positive=['reference', 0], negative=['segtext', 0], latent_image=['masked', 0], steps=20, start_at_step=0, end_at_step=10000, cfg=1, add_noise='enable', scheduler='simple'),
        'decode': node('VAEDecode', samples=['segment', 0], vae=['vae', 0]),
        'split': node('SplitImageWithAlpha', image=['decode', 0]),
        'scale': node('ImageScale', image=['split', 0]),
        'compose': node('SwarmImageCompositeMaskedColorCorrecting', source=['scale', 0], destination=['basepixels', 0], mask=['cropmask', 0]),
        'save': node('SaveImage', images=['compose', 0]),
    }


class GraphTests(unittest.TestCase):
    def test_locates_segment_not_base(self):
        out = c.analyze(fixture())
        self.assertEqual(out['segment_sampler_count'], 1)
        self.assertEqual(out['segments'][0]['sampler'], 'segment')

    def test_scope_and_model_connection(self):
        s = c.analyze(fixture())['segments'][0]
        self.assertEqual(s['lora_nodes_not_shared_with_base'], ['scoped'])
        self.assertEqual({l['inputs']['lora_name'] for l in s['lora_nodes']}, {'speed.safetensors', 'face.safetensors'})

    def test_missing_lora_connection_is_detected(self):
        g = fixture()
        g['segment']['inputs']['model'] = ['model', 0]
        s = c.analyze(g)['segments'][0]
        self.assertTrue(any(w.startswith('NO_LORA_NODE') for w in s['warnings']))
        self.assertFalse(s['lora_nodes'])

    def test_detached_lora_does_not_count(self):
        g = fixture()
        g['segment']['inputs']['model'] = ['global', 0]
        s = c.analyze(g)['segments'][0]
        self.assertEqual(s['lora_nodes_not_shared_with_base'], [])

    def test_conditioning_does_not_follow_reference_into_base(self):
        s = c.analyze(fixture())['segments'][0]
        self.assertEqual([n['id'] for n in s['conditioning_encoders']], ['segtext'])

    def test_reference_image_warning(self):
        g = fixture()
        self.assertTrue(any(w.startswith('REFERENCE_WITHOUT') for w in c.analyze(g)['segments'][0]['warnings']))
        g['segtext']['inputs']['images'] = ['crop', 0]
        self.assertFalse(any(w.startswith('REFERENCE_WITHOUT') for w in c.analyze(g)['segments'][0]['warnings']))

    def test_shipped_qwen_encoder_is_recognized_as_conditioning_leaf(self):
        graph = fixture()
        graph['segtext'] = {
            'class_type': 'SwarmQwenImage21SegmentEncode',
            'inputs': {'clip': ['global', 0], 'vae': ['vae', 0],
                       'image': ['crop', 0], 'image_1': ['basepixels', 0],
                       'prompt': 'person', 'negative_prompt': '',
                       'use_original_reference': True},
        }
        graph['segment']['inputs']['positive'] = ['segtext', 0]
        graph['segment']['inputs']['negative'] = ['segtext', 1]
        report = c.analyze(graph)['segments'][0]
        self.assertEqual([n['id'] for n in report['conditioning_encoders']], ['segtext'])
        self.assertEqual(report['conditioning_encoders'][0]['inputs']['image'], ['crop', 0])
        self.assertEqual(report['conditioning_encoders'][0]['inputs']['image_1'], ['basepixels', 0])

    def test_actual_start_step_and_disabled_noise(self):
        g = fixture()
        g['segment']['inputs'].update(start_at_step=20, add_noise='disable')
        s = c.analyze(g)['segments'][0]
        self.assertEqual(s['settings']['effective_step_count'], 0)
        self.assertIn('ZERO_EFFECTIVE_STEPS', s['warnings'])
        self.assertIn('SAMPLER_NOISE_DISABLED', s['warnings'])

    def test_graph_cycles_terminate(self):
        g = fixture()
        g['scoped']['inputs']['model'] = ['scoped', 0]
        self.assertEqual(c.analyze(g)['segment_sampler_count'], 1)

    def test_editor_graph_is_not_an_api_graph(self):
        self.assertIsNone(c.graph_or_none({'nodes': [], 'links': []}))

    def test_empty_deleted_history_can_capture_active_queue(self):
        responses = [
            {"system": {}}, {}, {"queue_running": [], "queue_pending": []},
            {}, {"queue_running": [[2, "live-prompt", fixture()]], "queue_pending": []},
        ]
        with patch.object(c, "get_json", side_effect=responses), patch.object(c.time, "sleep"):
            records, runtime = c.capture("http://127.0.0.1:7801/ComfyBackendDirect", wait=2)
        self.assertEqual(records[0]["prompt_id"], "live-prompt")
        self.assertEqual(records[0]["source"], "queue_running")

    def test_no_workflow_is_failure_not_pass(self):
        responses = [{"system": {}}, {}, {"queue_running": [], "queue_pending": []}]
        with patch.object(c, "get_json", side_effect=responses):
            with self.assertRaises(c.CaptureError):
                c.capture("http://127.0.0.1:7801/ComfyBackendDirect", wait=0)

    def test_history_and_queue_ordering(self):
        records = c.candidates({'p': {'prompt': [1, 'old', fixture()]}}, {'queue_running': [[2, 'new', fixture()]]})
        self.assertEqual([r[1] for r in records], ['new', 'old'])

    def test_credentials_and_media_removed_not_prompts(self):
        data = {'api_key': 'secret', 'image_base64': 'AAAA', 'prompt': 'actual trigger'}
        redacted = c.redact(data)
        self.assertNotIn('secret', json.dumps(redacted))
        self.assertNotIn('AAAA', json.dumps(redacted))
        self.assertEqual(redacted['prompt'], 'actual trigger')

    def test_bundle_and_original_graph_not_mutated(self):
        g = fixture()
        original = copy.deepcopy(g)
        report = c.analyze(g)
        path = c.write_bundle([{'graph': g, 'analysis': report, 'prompt_id': 'p', 'source': 'fixture'}], {}, PATH.parent)
        with zipfile.ZipFile(path) as z:
            self.assertEqual(len(z.namelist()), 4)
            self.assertEqual(json.loads(z.read('case-1-report.json'))['analysis']['segment_sampler_count'], 1)
        self.assertEqual(g, original)
        path.unlink()
        path.parent.rmdir()


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(('GET', self.path))
                paths = {
                    '/system_stats': {'system': {'python_version': 'fixture', 'argv': ['secret-not-to-be-captured']}},
                    '/history?max_items=8': {'p': {'prompt': [1, 'test-prompt', fixture()]}},
                    '/queue': {'queue_running': [], 'queue_pending': []},
                }
                if self.path not in paths:
                    self.send_error(404)
                    return
                raw = json.dumps(paths[self.path]).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_POST(self):
                calls.append(('POST', self.path))
                self.send_error(405)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
    def test_capture_uses_get_only_and_actual_history_graph(self):
        records, runtime = c.capture(self.url)
        self.assertEqual(records[0]['prompt_id'], 'test-prompt')
        self.assertEqual(records[0]['analysis']['segments'][0]['sampler'], 'segment')
        self.assertEqual([r[0] for r in self.calls], ['GET', 'GET', 'GET'])
        self.assertNotIn('argv', runtime['system'])
    def test_unknown_endpoint_fails_not_success(self):
        with self.assertRaises(c.CaptureError):
            c.get_json(self.url, 'not-a-route')
    def test_credentials_in_url_rejected(self):
        with self.assertRaises(c.CaptureError):
            c.get_json('http://user:pass@127.0.0.1', 'system_stats')
    def test_real_cli_creates_read_only_bundle(self):
        self.assertEqual(c.main(['--url', self.url]), 0)
        self.assertTrue(all(method == 'GET' for method, _ in self.calls))


if __name__ == '__main__':
    unittest.main()
