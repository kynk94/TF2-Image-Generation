import os
import re


def check_archive(path):
    if not os.path.exists(path):
        raise FileNotFoundError('The archive does not exist.')

    archive_name = re.search(r'[^./\\]+(?=\.)', path)
    if archive_name is None:
        raise NameError('archive file has no extension.')
    return archive_name.group()
