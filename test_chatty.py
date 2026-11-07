import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import chatty


class Handler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.calls.append((self.path, dict(self.headers), body))
        if self.path.endswith('/responses'):
            result = {'output': [{'content': [{'type': 'output_text', 'text': 'response'}]}]}
        elif self.path.endswith('/messages'):
            result = {'content': [{'type': 'text', 'text': 'response'}]}
        else:
            result = {'choices': [{'message': {'content': 'response'}}]}
        data = json.dumps(result).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(data)

    def log_message(self, *args): pass


class Tests(unittest.TestCase):
    def test_protocols_and_images(self):
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / 'image.png'; image.write_bytes(b'test image')
                for protocol in chatty.PROTOCOLS:
                    c = {'url': f'http://127.0.0.1:{server.server_port}/v1', 'protocol': protocol, 'model': 'test', 'api_key': 'secret', 'reasoning': 'high'}
                    self.assertEqual(chatty.request_chat(c, [chatty.user_message('hello', [image], protocol)]), 'response')
                    path, headers, body = Handler.calls[-1]
                    self.assertEqual(path, '/v1/' + {'responses': 'responses', 'chat': 'chat/completions', 'messages': 'messages'}[protocol])
                    headers = {k.lower(): v for k, v in headers.items()}
                    self.assertEqual(headers.get('x-api-key') if protocol == 'messages' else headers.get('authorization'), 'secret' if protocol == 'messages' else 'Bearer secret')
                    self.assertEqual(body['model'], 'test')
                    self.assertEqual(len(body['input' if protocol == 'responses' else 'messages'][0]['content']), 2)
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_mantle_and_config(self):
        with patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': 'key', 'AWS_REGION': 'us-east-1'}, clear=True):
            c = chatty.environment_connections()['env:Bedrock Mantle']
            self.assertEqual(c['url'], 'https://bedrock-mantle.us-east-1.api.aws/v1')
            self.assertEqual(c['api_key'], 'key')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'connections.json'
            chatty.save_config(path, {'test': c})
            self.assertEqual(chatty.load_config(path)['test'], c)


if __name__ == '__main__': unittest.main()
