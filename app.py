"""WSGI entrypoint for Vercel.

Vercel's Python runtime loads one top-level `app` and routes every request to it,
so this serves the static page as well as the engine API. Both go through the same
`dispatch()` the local server uses, so the deployment cannot drift from
`bash run.sh site`.

Deliberately plain WSGI: no web framework, which keeps the function bundle to
numpy + python-chess and the cold start short.
"""
import json
from http import HTTPStatus
from pathlib import Path
from fastchess.serverless import MAX_BODY, engine
from fastchess.site_server import dispatch

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / 'public'
CONTENT_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
                 '.css': 'text/css', '.json': 'application/json',
                 '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon'}


def reply(start_response, status, body, content_type, cache='no-store'):
    start_response(f'{status} {HTTPStatus(status).phrase}',
                   [('Content-Type', content_type),
                    ('Content-Length', str(len(body))),
                    ('Cache-Control', cache)])
    return [body]


def as_json(start_response, status, payload):
    return reply(start_response, status, json.dumps(payload).encode(), 'application/json')


def app(environ, start_response):
    method = environ.get('REQUEST_METHOD', 'GET')
    path = environ.get('PATH_INFO', '/') or '/'

    if path.startswith('/api/'):
        if method != 'POST':
            return as_json(start_response, 405, {'error': 'use POST'})
        try:
            length = int(environ.get('CONTENT_LENGTH') or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY:
            return as_json(start_response, 413, {'error': 'request too large'})
        try:
            raw = environ['wsgi.input'].read(length) if length else b'{}'
            request = json.loads(raw or b'{}')
            if not isinstance(request, dict):
                raise ValueError('body must be a JSON object')
        except (ValueError, json.JSONDecodeError) as error:
            return as_json(start_response, 400, {'error': str(error)})
        status, payload = dispatch(engine(), path[len('/api/'):], request)
        return as_json(start_response, status, payload)

    if method not in ('GET', 'HEAD'):
        return as_json(start_response, 405, {'error': 'method not allowed'})
    name = 'index.html' if path in ('/', '/index.html') else path.lstrip('/')
    target = (PUBLIC / name).resolve()
    # Never serve outside public/, whatever the request path contains.
    if target.parent != PUBLIC.resolve() or not target.is_file():
        return reply(start_response, 404, b'Not found', 'text/plain; charset=utf-8')
    body = b'' if method == 'HEAD' else target.read_bytes()
    return reply(start_response, 200, body,
                 CONTENT_TYPES.get(target.suffix, 'application/octet-stream'))


# Some WSGI hosts look for `application` instead.
application = app
