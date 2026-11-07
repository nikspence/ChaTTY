"""ChaTTY: a small terminal client for compatible LLM HTTP APIs."""
import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

DEFAULT_CONFIG = Path(os.getenv('CHATTY_CONFIG', Path.home() / '.config' / 'chatty' / 'connections.json'))
PROTOCOLS = ('responses', 'chat', 'messages')

# name, protocol, URL variables, key variables, default URL, model variable
PROVIDERS = [
    ('OpenAI', 'responses', ('OPENAI_BASE_URL', 'OPENAI_API_BASE'), ('OPENAI_API_KEY',), 'https://api.openai.com/v1', 'OPENAI_MODEL'),
    ('Anthropic', 'messages', ('ANTHROPIC_BASE_URL',), ('ANTHROPIC_API_KEY',), 'https://api.anthropic.com/v1', 'ANTHROPIC_MODEL'),
    ('LiteLLM', 'chat', ('LITELLM_BASE_URL', 'LITELLM_API_BASE', 'LITELLM_URL'), ('LITELLM_API_KEY', 'LITELLM_MASTER_KEY'), '', 'LITELLM_MODEL'),
    ('Ollama', 'chat', ('OLLAMA_HOST', 'OLLAMA_BASE_URL'), ('OLLAMA_API_KEY',), 'http://localhost:11434/v1', 'OLLAMA_MODEL'),
    ('llama.cpp', 'chat', ('LLAMA_CPP_BASE_URL', 'LLAMA_CPP_URL', 'LLAMACPP_BASE_URL'), ('LLAMA_CPP_API_KEY', 'LLAMACPP_API_KEY'), '', 'LLAMA_CPP_MODEL'),
    ('OpenRouter', 'chat', ('OPENROUTER_BASE_URL',), ('OPENROUTER_API_KEY',), 'https://openrouter.ai/api/v1', 'OPENROUTER_MODEL'),
    ('Groq', 'chat', ('GROQ_BASE_URL',), ('GROQ_API_KEY',), 'https://api.groq.com/openai/v1', 'GROQ_MODEL'),
    ('Together', 'chat', ('TOGETHER_BASE_URL',), ('TOGETHER_API_KEY',), 'https://api.together.xyz/v1', 'TOGETHER_MODEL'),
    ('Mistral', 'chat', ('MISTRAL_BASE_URL',), ('MISTRAL_API_KEY',), 'https://api.mistral.ai/v1', 'MISTRAL_MODEL'),
    ('DeepSeek', 'chat', ('DEEPSEEK_BASE_URL',), ('DEEPSEEK_API_KEY',), 'https://api.deepseek.com/v1', 'DEEPSEEK_MODEL'),
    ('xAI', 'chat', ('XAI_BASE_URL',), ('XAI_API_KEY',), 'https://api.x.ai/v1', 'XAI_MODEL'),
]


def first(names):
    return next((os.environ[n] for n in names if os.environ.get(n)), '')


def environment_connections():
    found = {}
    for name, protocol, hosts, keys, default, modelvar in PROVIDERS:
        url, key, model = first(hosts), first(keys), os.getenv(modelvar, '')
        if url or key or model:
            found['env:' + name] = dict(url=url or default, api_key=key, protocol=protocol, model=model, reasoning='')
    key = first(('AWS_BEARER_TOKEN_BEDROCK', 'BEDROCK_API_KEY'))
    url = first(('BEDROCK_MANTLE_BASE_URL', 'BEDROCK_MANTLE_URL'))
    region = first(('AWS_REGION', 'AWS_DEFAULT_REGION'))
    if key or url:
        found['env:Bedrock Mantle'] = dict(url=url or (f'https://bedrock-mantle.{region}.api.aws/v1' if region else ''), api_key=key, protocol='responses', model=os.getenv('BEDROCK_MODEL', ''), reasoning='')
    return found


def load_config(path):
    if not path.exists(): return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('connections', {}), dict):
        raise ValueError('Config must contain a connections mapping')
    result = data.get('connections', {})
    if any(not isinstance(k, str) or not isinstance(v, dict) for k, v in result.items()):
        raise ValueError('Each connection must be a named mapping')
    return result


def save_config(path, connections):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'connections': connections}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def endpoint(url, protocol):
    url = url.rstrip('/')
    suffix = {'responses': '/responses', 'chat': '/chat/completions', 'messages': '/messages'}[protocol]
    if url.endswith(suffix): return url
    if '11434' in url and '/' not in url.split('://', 1)[-1]: url += '/v1'
    return url + suffix


def image_block(path, protocol):
    mime = mimetypes.guess_type(str(path))[0]
    if not mime or not mime.startswith('image/'): raise ValueError(f'Unknown image type: {path}')
    encoded = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    if protocol == 'messages': return {'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': encoded}}
    if protocol == 'responses': return {'type': 'input_image', 'image_url': f'data:{mime};base64,{encoded}'}
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}}


def user_message(text, images, protocol):
    if not images: return {'role': 'user', 'content': text}
    return {'role': 'user', 'content': [{'type': 'input_text' if protocol == 'responses' else 'text', 'text': text}, *(image_block(p, protocol) for p in images)]}


def request_chat(c, history):
    protocol = c.get('protocol', 'responses')
    if protocol not in PROTOCOLS: raise ValueError('Invalid protocol')
    if not c.get('url') or not c.get('model'): raise ValueError('A URL and model are required')
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    key, reasoning = c.get('api_key', ''), c.get('reasoning', '')
    if protocol == 'messages':
        if key: headers['x-api-key'] = key
        headers['anthropic-version'] = '2023-06-01'
        body = {'model': c['model'], 'messages': history, 'max_tokens': 4096}
        if reasoning: body.update(thinking={'type': 'adaptive'}, output_config={'effort': reasoning})
    else:
        if key: headers['Authorization'] = 'Bearer ' + key
        body = {'model': c['model'], 'input' if protocol == 'responses' else 'messages': history}
        if reasoning:
            if protocol == 'responses': body['reasoning'] = {'effort': reasoning}
            else: body['reasoning_effort'] = reasoning
    req = urllib.request.Request(endpoint(c['url'], protocol), data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=300) as response: data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'HTTP {exc.code}: {exc.read().decode("utf-8", errors="replace")}') from exc
    except urllib.error.URLError as exc:
        raise ValueError(f'Connection failed: {exc.reason}') from exc
    try:
        if protocol == 'chat':
            result = data['choices'][0]['message']['content']
            return ''.join(x.get('text', '') for x in result) if isinstance(result, list) else (result or '')
        if protocol == 'messages': return ''.join(x.get('text', '') for x in data['content'] if x.get('type') == 'text')
        return data.get('output_text') or ''.join(x.get('text', '') for item in data['output'] for x in item.get('content', []) if x.get('type') == 'output_text')
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f'Unexpected response: {json.dumps(data)[:1000]}') from exc


def choose(connections):
    names = list(connections)
    if not names: print('No connections available.'); return None
    for i, name in enumerate(names, 1):
        c = connections[name]; print(f'  {i}. {name} ({c.get("protocol", "responses")}, {c.get("model") or "no model"})')
    choice = input('Choose number (Enter cancels): ').strip()
    if not choice: return None
    if not choice.isdigit() or not 1 <= int(choice) <= len(names): raise ValueError('Invalid selection')
    return names[int(choice) - 1]


def edit(c=None):
    c = dict(c or {})
    for field, label in [('url', 'Endpoint URL'), ('api_key', 'API key (plain text)'), ('protocol', 'Protocol (responses/chat/messages)'), ('model', 'Model'), ('reasoning', 'Reasoning effort (optional)')]:
        old = c.get(field, 'responses' if field == 'protocol' else '')
        display = 'set' if field == 'api_key' and old else old
        value = input(f'{label}' + (f' [{display}]' if display else '') + ': ').strip()
        c[field] = '' if value == ':clear' else value or old
    if c['protocol'] not in PROTOCOLS: raise ValueError('Invalid protocol')
    return c


def manager(saved, path):
    while True:
        action = input('\nConnection manager: [a]dd [e]dit [d]elete [l]ist [b]ack > ').strip().lower()
        try:
            if action == 'b': return
            if action == 'l':
                for name, c in {**environment_connections(), **saved}.items(): print(f'{name}: {c.get("protocol")} {c.get("url")} {c.get("model", "")}')
            elif action == 'a':
                name = input('Name: ').strip()
                if not name or name.startswith('env:') or name in saved: raise ValueError('Name must be new, nonempty, and not start with env:')
                saved[name] = edit(); save_config(path, saved)
            elif action in ('e', 'd'):
                name = choose(saved)
                if name and action == 'e': saved[name] = edit(saved[name]); save_config(path, saved)
                elif name and input(f'Delete {name}? [y/N] ').strip().lower() == 'y': del saved[name]; save_config(path, saved)
        except (ValueError, OSError) as exc: print(f'Error: {exc}')


def tui(saved, path):
    name, active, history, images = None, {}, [], []
    print('ChaTTY — LLM terminal chat\n/connect /connections /model /reasoning /new /image /help /quit')
    while True:
        try:
            line = input(f'\n{name or "no connection"}> ').strip()
            if not line: continue
            cmd, _, arg = line.partition(' ')
            if cmd in ('/quit', '/exit'): return
            if cmd == '/help': print('/connect selects a connection; /connections manages saved connections; /model NAME; /reasoning LEVEL or off; /new clears chat; /image PATH attaches an image; /quit exits.')
            elif cmd == '/connections': manager(saved, path)
            elif cmd == '/connect':
                options = {**environment_connections(), **saved}; selected = choose(options)
                if selected: name, active, history, images = selected, dict(options[selected]), [], []
            elif cmd == '/new': history, images = [], []; print('New chat started.')
            elif cmd in ('/model', '/reasoning'):
                if not name: print('Select a connection first.'); continue
                value = arg or input(cmd[1:] + ': ').strip()
                active[cmd[1:]] = '' if value.lower() == 'off' else value
            elif cmd == '/image':
                if not name: print('Select a connection first.'); continue
                image_block(arg.strip('"'), active.get('protocol', 'responses')); images.append(arg.strip('"')); print('Attached to next message.')
            elif line.startswith('/'): print('Unknown command. Type /help.')
            elif not name: print('Use /connect to select a connection.')
            else:
                trial = history + [user_message(line, images, active.get('protocol', 'responses'))]
                answer = request_chat(active, trial)
                history = trial + [{'role': 'assistant', 'content': answer}]; images = []
                print('\n' + answer)
        except (ValueError, OSError) as exc: print(f'Error: {exc}', file=sys.stderr)
        except (EOFError, KeyboardInterrupt): print(); return


def main():
    p = argparse.ArgumentParser(description='ChaTTY: chat with compatible LLM endpoints')
    p.add_argument('prompt', nargs='?', help='One-shot prompt; omit for TUI')
    p.add_argument('-c', '--connection', help='Saved connection name or env:NAME')
    p.add_argument('--url', help='Base or complete endpoint URL')
    p.add_argument('--api-key'); p.add_argument('--protocol', choices=PROTOCOLS)
    p.add_argument('-m', '--model'); p.add_argument('--reasoning')
    p.add_argument('-i', '--image', action='append', default=[], help='Image path; repeatable')
    p.add_argument('-o', '--output', help='Write response to a UTF-8 file')
    p.add_argument('-save', '--save', metavar='NAME'); p.add_argument('--delete', metavar='NAME')
    p.add_argument('--list', action='store_true'); p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    a = p.parse_args()
    try:
        saved = load_config(a.config)
        if a.delete:
            if a.delete not in saved: raise ValueError(f'No saved connection: {a.delete}')
            del saved[a.delete]; save_config(a.config, saved); print(f'Deleted {a.delete}.', file=sys.stderr)
        if a.list:
            for name, c in {**environment_connections(), **saved}.items(): print(f'{name}\t{c.get("protocol", "responses")}\t{c.get("url", "")}\t{c.get("model", "")}')
        options = {**environment_connections(), **saved}
        if a.connection and a.connection not in options: raise ValueError(f'Unknown connection: {a.connection}')
        c = dict(options.get(a.connection, {}))
        for field in ('url', 'api_key', 'protocol', 'model', 'reasoning'):
            value = getattr(a, field)
            if value is not None: c[field] = '' if field == 'reasoning' and value.lower() == 'off' else value
        c.setdefault('protocol', 'responses')
        if a.save:
            if a.save.startswith('env:'): raise ValueError('Saved names cannot start with env:')
            if not c.get('url'): raise ValueError('A URL is required to save a connection')
            saved[a.save] = c; save_config(a.config, saved); print(f'Saved {a.save} to {a.config}.', file=sys.stderr)
        if a.prompt is not None:
            answer = request_chat(c, [user_message(a.prompt, a.image, c['protocol'])])
            if a.output: Path(a.output).write_text(answer, encoding='utf-8')
            else: print(answer)
        elif a.image or a.output: raise ValueError('--image and --output require a prompt')
        elif not (a.save or a.delete or a.list): tui(saved, a.config)
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr); return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
