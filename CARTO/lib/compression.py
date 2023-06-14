import json
import zstandard
import collections
import io
import tarfile


def write_compressed_json(x, path):
    cctx = zstandard.ZstdCompressor()
    with open(path, "wb") as raw_fh:
        with cctx.stream_writer(raw_fh) as zst_fh:
            zst_fh.write(json.dumps(x, sort_keys=True, indent=2).encode())


def read_compressed_json(path):
    cctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as raw_fh:
        with cctx.stream_reader(raw_fh) as zst_fh:
            bytes_ = zst_fh.read()
            str_ = bytes_.decode()
            x = json.loads(str_, object_pairs_hook=collections.OrderedDict)
            return x


def extract_compressed_tarfile(tarfile_path, dst_dir):
    cctx = zstandard.ZstdDecompressor()
    with open(tarfile_path, "rb") as raw_fh:
        with cctx.stream_reader(raw_fh) as zst_fh:
            tarfile_buf = zst_fh.read()

    with io.BytesIO(tarfile_buf) as raw_fh:
        with tarfile.TarFile(fileobj=raw_fh) as tar:
            members = tar.getmembers()
            for member in members:
                if not member.isfile():
                    continue
                data = tar.extractfile(member).read()
                assert member.name[0] != "/"
                member_path = dst_dir / member.path
                parent_dir = member_path.parent
                parent_dir.mkdir(parents=True, exist_ok=True)
                with open(member_path, "wb") as f:
                    f.write(data)
