import glob
import json
import os
import platform
import shutil
from urllib.parse import unquote


def find_files_with_suffix(root_dir, suffix):
    search_pattern = os.path.join(root_dir, '**', f'*.{suffix}')
    found_files = glob.glob(search_pattern, recursive=True)
    return found_files

def delete_directory(path):
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)


def generate_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def dump_file(full_path, data):
    full_path = '\\\\?\\' + full_path
    if not full_path.endswith(".json"):
        full_path += ".json"

    with open(full_path, 'w') as data_file:
        json.dump(data, data_file, indent=4, sort_keys=True)


def move_directory(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.move(s, d)


def normalize_string(str):
    str = str \
        .replace(' ', '_') \
        .replace('-', '') \
        .lower()
    return str


def normilize_path(path):
    return os.path.normpath(path)


def uri_to_path(uri):
    import os
    from urllib.parse import urlparse
    path = ''
    if platform.system() == "Windows":
        path = unquote(urlparse(uri).path).replace('/', '')
    elif platform.system() == "Linux":
        path = unquote(urlparse(uri).path).replace('//', '/')
    return path


def get_file_name_from_full_path(file):
    system = platform.system()
    if system == "Linux":
        pos = file.rfind('/') + 1
        file_name = file[pos:]
    elif system == "Windows":
        pos = file.rfind('\\') + 1
        file_name = file[pos:]
    return file_name


def move_file(src, dst_dir):
    shutil.move(src, dst_dir)


def copy_tree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copyfile(s, d)
