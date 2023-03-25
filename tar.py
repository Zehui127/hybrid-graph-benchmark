import os
import tarfile


def create_tar_file(tar_file_path, files_to_tar):
    with tarfile.open(tar_file_path, 'w') as tar:
        for file_path in files_to_tar:
            tar.add(file_path, arcname=os.path.basename(file_path))


tar_file_path = '/Users/lizehui/Desktop/workspace/hypergraph-benchmarks/output.tar'
files_to_tar = '/Users/lizehui/Desktop/workspace/hypergraph-benchmarks/grand_ten'

create_tar_file(tar_file_path, files_to_tar)
