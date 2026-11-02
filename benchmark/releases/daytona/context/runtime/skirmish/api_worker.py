"""Trusted, killable HTTP request worker. Secrets arrive on stdin, never argv/files."""
import json
import sys
from .providers import perform_request

if __name__ == '__main__':
    data = json.load(sys.stdin)
    try:
        result = perform_request(**data)
        print(json.dumps(dict(ok=True,result=result)))
    except Exception as error:
        print(json.dumps(dict(ok=False,error=f'{type(error).__name__}: {str(error)[:200]}')))
