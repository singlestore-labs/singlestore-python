#!/usr/bin/env python
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
import venv
from multiprocessing import Process
from multiprocessing import Semaphore
from typing import Any
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from IPython.core.magic import register_line_cell_magic
from IPython.core.magic import register_line_magic
from packaging.requirements import Requirement

from ..management import workspace

LOCAL_ENV_DIR = 'environment'
OLD_LOCAL_ENV_DIR = '.old-environments'
REMOTE_ENV_DIR = 'environments'
UPLOAD_RETRIES = 3
DOWNLOAD_RETRIES = 3
MAX_CHUNK_SIZE = 100 * int(1e6)
MAX_DOWNLOAD_PROCS = 5
MAX_UPLOAD_PROCS = 5


def _upload_file(
    src_path: str,
    target_path: str,
    semaphore: Any,
) -> None:
    semaphore.acquire()
    try:
        n = UPLOAD_RETRIES
        while True:
            try:
                workspace.get_stage().upload_file(src_path, target_path, overwrite=True)
                break
            except Exception as exc:
                n -= 1
                if not n:
                    raise RuntimeError('failed to upload {src_path}') from exc
    finally:
        semaphore.release()


def _download_env_chunk(
    src_path: str,
    env_path: str,
    semaphore: Any,
) -> None:
    semaphore.acquire()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(src_path))
            n = DOWNLOAD_RETRIES
            while True:
                try:
                    workspace.get_stage().download_file(
                        src_path,
                        local_path,
                        overwrite=True,
                    )
                    break
                except Exception as exc:
                    n -= 1
                    if not n:
                        raise RuntimeError('failed to download {src_path}') from exc
            with tarfile.open(local_path, 'r') as z:
                z.extractall(env_path)
    finally:
        semaphore.release()


def _env_from_stage(name: str, verbose: bool = False) -> None:
    """Install an env from Stage."""
    pyenv = os.path.join(os.getcwd(), LOCAL_ENV_DIR)
    env_dir = f'{REMOTE_ENV_DIR}/{name}/'

    if os.path.exists(pyenv):
        shutil.move(pyenv, f'{OLD_LOCAL_ENV_DIR}/{str(uuid.uuid4())}')

    os.makedirs(pyenv, exist_ok=True)

    print('Loading environment from stage...')

    # Create directory structure
    with tempfile.TemporaryDirectory() as tmp:
        workspace.get_stage().download_file(
            f'{env_dir}dirs.tar',
            f'{tmp}/dirs.tar',
            overwrite=True,
        )
        tarfile.open(f'{tmp}/dirs.tar', 'r').extractall(env_dir)

    # Load each file from manifest
    manifest = str(
        workspace.get_stage().download_file(
            f'{env_dir}manifest.txt',
            encoding='utf-8',
        ) or '',
    )

    semaphore = Semaphore(MAX_DOWNLOAD_PROCS)
    processes = []
    for chunk_name in [x.strip() for x in manifest.split('\n') if x.strip()]:
        p = Process(
            target=_download_env_chunk,
            args=(f'{env_dir}{chunk_name}', pyenv, semaphore),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _use_env(
    name: str,
    packages: Optional[List[str]] = None,
    verbose: bool = False,
    force_reinstall: bool = False,
    upload: bool = True,
) -> None:
    """
    Install a Python environment.

    Parameters
    ----------
    name : str
        Name of the environment
    packages : list[str], optional
        List of package versions to install
    verbose : bool, optional
        Run pip / conda in verbose mode

    """
    if not name:
        raise ValueError('name is empty')

    pyenv = os.path.join(os.getcwd(), LOCAL_ENV_DIR)
    verb = ['-v'] if verbose else []

    # Add env to Python path
    version = f'python{sys.version_info.major}.{sys.version_info.minor}'
    site_packages = os.path.join(pyenv, 'lib', version, 'site-packages')
    if sys.path[0] != site_packages:
        sys.path.insert(0, site_packages)

    # Add env bin to system path
    path = os.environ['PATH'].split(':')
    if path[0] != f'{pyenv}/bin':
        os.environ['PATH'] = f'{pyenv}/bin:{os.environ["PATH"]}'

    env_dir = f'{REMOTE_ENV_DIR}/{name}/'

    # If environment exists in stage, use it
    if not force_reinstall and workspace.get_stage().exists(env_dir):
        return _env_from_stage(name, verbose)

    if not packages:
        raise RuntimeError('no packages specified and no existing environment found')

    print('Creating new environment...')

    if os.path.exists(pyenv):
        os.rename(pyenv, f'{OLD_LOCAL_ENV_DIR}/{str(uuid.uuid4())}')

    builder = venv.EnvBuilder(
        system_site_packages=False,
        clear=True,
        symlinks=True,
        upgrade=False,
        with_pip=True,
        prompt=None,
    )

    builder.create(pyenv)

    if packages:
        rc = subprocess.call(['pip', 'install'] + verb + packages)
        if rc != 0:
            raise RuntimeError('pip installation failed')

    if not upload:
        return

    if workspace.get_stage().exists(env_dir):
        workspace.get_stage().removedirs(env_dir)

    if not workspace.get_stage().exists(f'{REMOTE_ENV_DIR}/'):
        workspace.get_stage().mkdir(f'{REMOTE_ENV_DIR}/')

    workspace.get_stage().mkdir(env_dir)

    # Upload env in chunks
    with tempfile.TemporaryDirectory() as tmp:

        dirs_tar = tarfile.open(f'{tmp}/dirs.tar', 'w')
        manifest = open(f'{tmp}/manifest.txt', 'w')

        with open(f'{tmp}/requirements.txt', 'wb') as req:
            req.write(subprocess.check_output(['pip', 'freeze']))

        processes = []

        n = 1
        chunk_size = 0

        local_chunk = f'{tmp}/chunk{n}.tar'
        remote_chunk = f'{env_dir}chunk{n}.tar'

        manifest.write(os.path.basename(remote_chunk) + '\n')

        chunk = tarfile.open(local_chunk, mode='w')

        semaphore = Semaphore(MAX_UPLOAD_PROCS)

        for root, dirs, files in os.walk(pyenv, followlinks=False):

            for d in dirs:
                dpath = os.path.join(root, d)
                dirs_tar.add(dpath, arcname=dpath[len(pyenv) + 1:], recursive=False)

            for f in files:
                fpath = os.path.join(root, f)

                path_size = os.path.getsize(fpath)

                if chunk_size + path_size > MAX_CHUNK_SIZE:
                    chunk_size = 0
                    chunk.close()

                    p = Process(
                        target=_upload_file,
                        args=(local_chunk, remote_chunk, semaphore),
                    )
                    p.start()
                    processes.append(p)

                    n += 1

                    local_chunk = f'{tmp}/chunk{n}.tar'
                    remote_chunk = f'{env_dir}chunk{n}.tar'

                    manifest.write(os.path.basename(remote_chunk) + '\n')
                    chunk = tarfile.open(local_chunk, mode='w')

                chunk_size += path_size
                chunk.add(fpath, arcname=fpath[len(pyenv) + 1:], recursive=False)

        chunk.close()

        p = Process(target=_upload_file, args=(local_chunk, remote_chunk, semaphore))
        p.start()
        processes.append(p)

        manifest.close()
        dirs_tar.close()

        workspace.get_stage().upload_file(
            f'{tmp}/requirements.txt',
            stage_path=f'{env_dir}requirements.txt',
            overwrite=True,
        )
        workspace.get_stage().upload_file(
            f'{tmp}/dirs.tar',
            stage_path=f'{env_dir}dirs.tar',
            overwrite=True,
        )
        workspace.get_stage().upload_file(
            f'{tmp}/manifest.txt',
            stage_path=f'{env_dir}manifest.txt',
            overwrite=True,
        )

        for p in processes:
            p.join()


def _rm_env(name: str) -> None:
    """Delete an environment from Stage."""
    if not name:
        raise ValueError('no environment name given')

    if workspace.get_stage().exists(f'{REMOTE_ENV_DIR}/{name}/'):
        workspace.get_stage().removedirs(f'{REMOTE_ENV_DIR}/{name}/')


#
# Magic commands
#


def compare_req(req_a: Requirement, req_b: Requirement) -> bool:
    """See if requirements reconcile to the same version."""
    if req_a.name != req_b.name:
        return False
    if req_a.url != req_b.url:
        return False
    if req_a.specifier == '' or req_b.specifier == '':
        return True
    if '==' in str(req_a.specifier):
        return str(req_a.specifier).split('==', 1)[-1] in req_b.specifier
    elif '==' in str(req_b.specifier):
        return str(req_b.specifier).split('==', 1)[-1] in req_a.specifier
    return False


def compare_req_list(reqs_a: Set[Requirement], reqs_b: Set[Requirement]) -> bool:
    """Compeare sets of requirements to see if they match."""
    if len(reqs_a) != len(reqs_b):
        return False
    for a, b in zip(
            sorted(reqs_a, key=lambda x: x.name),
            sorted(reqs_b, key=lambda x: x.name),
    ):
        if not compare_req(a, b):
            return False
    return True


def check_requirements(pip_install: str, requirements_txt: str) -> bool:
    """Compare a `pip install` call to a requirements.txt file to see if they match."""
    pip = {y.name: y for y in {Requirement(x) for x in pip_install.split() if x}}
    req = {y.name: y for y in {Requirement(x) for x in requirements_txt.split() if x}}
    for req_name in list(req.keys()):
        if req_name not in pip:
            del req[req_name]
    return compare_req_list(set(pip.values()), set(req.values()))


def _get_options(line: str) -> Tuple[str, List[str], List[str]]:
    """Split env name, options, and packages."""
    line = line.strip()
    opts = re.split(r'\s+', line)
    options = [x for x in opts if x and x.startswith('--')]
    packages = [x for x in opts if x and not x.startswith('--')]
    if packages:
        env_name = packages.pop(0)
    else:
        env_name = ''
    if not env_name:
        raise ValueError('no environment name given')
    return env_name, options, packages


@register_line_cell_magic
def use_env(line: str, cell: Optional[str] = None) -> None:
    """Magic for using a Python environment."""
    env_name, options, packages = _get_options(line)

    if cell:
        req_path = f'{REMOTE_ENV_DIR}/{env_name}/requirements.txt'
        if workspace.get_stage().exists(req_path):
            req_txt = str(
                workspace.get_stage().download_file(
                    req_path,
                    encoding='utf-8',
                ) or '',
            )

            if not check_requirements(cell, req_txt):
                options.append('--force')

        packages += [x for x in re.split(r'\s+', cell) if x]

    _use_env(
        env_name,
        packages,
        force_reinstall='--force' in options,
        upload='--no-upload' not in options,
    )


@register_line_magic
def rm_env(line: str) -> None:
    """Magic for deleting a Python environment."""
    if not line.strip():
        print('no env name given')
        return
    _rm_env(line.strip())


@register_line_magic
def list_env(line: str) -> None:
    """Magic for listing packages in an environment."""
    if '-v' in line or '--verbose' in line:
        subprocess.call(['pip', 'list', '-v'])
        return
    subprocess.call(['pip', 'list'])
