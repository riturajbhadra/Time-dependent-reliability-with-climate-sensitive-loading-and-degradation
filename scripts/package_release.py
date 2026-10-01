"""Build a source distribution with documentation, tests and data."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('src/wind_reliability','scripts','tests','config','docs','data','.github')
FILES = ('README.md','main.py','pyproject.toml','requirements.txt',
         'requirements-reproducible.txt','.gitignore')


def public_files():
    paths = [ROOT/name for name in FILES]
    for name in DIRECTORIES:
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc','.pyo')
                     and not any(part.endswith('.egg-info') for part in p.parts))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/wind_reliability.zip')
    args = parser.parse_args()
    paths = public_files()
    manifest = {str(p.relative_to(ROOT)).replace('\\','/'):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p,arcname=p.relative_to(ROOT))
        archive.writestr('release_manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
    print(f'{len(paths)} files packaged: {args.output}')


if __name__ == '__main__':
    main()
