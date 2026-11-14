"""Run: cd dspy-agent && python -m app.dev_chat (isolated local testing only)."""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8011)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    load_dotenv(root.parent / 'lab' / '1' / '.env')
    key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not key:
        parser.error('Set GEMINI_API_KEY or GOOGLE_API_KEY in .env')
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    os.environ['KB_ALLOW_UNREVIEWED'] = '1'
    os.environ['ENABLE_RESPONSE_GENERATOR'] = '1'
    os.environ['LOCAL_CHAT_INSPECTOR'] = '1'
    for name in ('LINE_CHANNEL_ACCESS_TOKEN', 'LINE_CHANNEL_SECRET', 'LINE_USER_HMAC_SECRET'):
        os.environ[name] = ''
    import dspy
    import uvicorn
    dspy.configure(lm=dspy.LM(model=os.getenv('GEMINI_MODEL', 'gemini/gemini-3.1-flash-lite'),
                             api_key=key, cache=False, timeout=45, num_retries=0))
    print(f'Open http://127.0.0.1:{args.port}/dev/chat', flush=True)
    print('Real Gemini | temporary database | LINE disabled | local inspector enabled', flush=True)
    uvicorn.run('app.main:app', host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
