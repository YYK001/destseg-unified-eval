"""Run on Kaggle only: download official DTD and extract images without overwriting."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request

URL='https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    root=args.directory.resolve()
    root.mkdir(parents=True,exist_ok=True)
    images=root/'dtd/images'
    existing=list(images.glob('*/*.jpg'))
    if len(existing)==5640 and len({p.parent for p in existing})==47:
        print(f'DTD already present: {images}',flush=True)
        return
    if images.exists():
        raise FileExistsError(f'Incomplete existing DTD directory; inspect or use a new --directory: {images}')
    archive=root/'dtd-r1.0.1.tar.gz'
    if not archive.exists():
        partial=archive.with_suffix('.gz.part')
        with urllib.request.urlopen(URL,timeout=120) as response,partial.open('wb') as output:
            total=int(response.headers.get('Content-Length',0))
            if total and shutil.disk_usage(root).free<total+256*1024**2:
                raise RuntimeError('Insufficient download space')
            done=0; last=0
            while True:
                block=response.read(4*1024**2)
                if not block: break
                output.write(block); done+=len(block)
                if time.monotonic()-last>5:
                    print(f'DTD download: {done/1024**2:.1f} / {total/1024**2:.1f} MiB',flush=True)
                    last=time.monotonic()
        if total and done!=total:
            raise RuntimeError('Incomplete DTD archive')
        partial.replace(archive)
    with tarfile.open(archive,'r:gz') as data:
        selected=[]
        for member in data.getmembers():
            path=PurePosixPath(member.name)
            if path.parts[:2]!=('dtd','images') or not member.isfile():
                continue
            if path.is_absolute() or '..' in path.parts or len(path.parts)!=4 or path.suffix!='.jpg':
                raise ValueError(f'Unexpected DTD path: {member.name}')
            selected.append(member)
        if len(selected)!=5640 or len({m.name for m in selected})!=5640:
            raise ValueError('Expected 5640 distinct DTD images')
        if shutil.disk_usage(root).free<sum(m.size for m in selected)+256*1024**2:
            raise RuntimeError('Insufficient extraction space')
        for i,member in enumerate(selected,1):
            target=root.joinpath(*PurePosixPath(member.name).parts)
            if not target.resolve().is_relative_to(root): raise ValueError(member.name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with data.extractfile(member) as source,target.open('xb') as out:
                shutil.copyfileobj(source,out)
            if i%1000==0 or i==5640: print(f'DTD extracted: {i}/5640',flush=True)
    print(f'DTD_ROOT = {images}',flush=True)
    print('Archive retained. No model training was started.',flush=True)


if __name__=='__main__':
    main()
