import os
import struct


def read_index(filename):
    index = []
    with open(filename) as index_file:
        while True:
            offset = index_file.read(8)
            if not offset:
                break
            offset, = struct.unpack('<Q', offset)
            index.append(offset)
    return index


class IndexedFileWriter(object):
    def __init__(self, path):
        self.f = open(path, 'wb')
        self.index_f = open(path + '.index', 'wb')

    def append(self, record):
        offset = self.f.tell()
        self.f.write(record)
        self.index_f.write(struct.pack('<Q', offset))

    def close(self):
        self.f.close()
        self.index_f.close()


class IndexedFileReader(object):
    def __init__(self, path):
        self.f = open(path, 'rb')

        self.index = read_index(path + '.index')
        self.lengths = [
            end - start
            for start, end in zip([0] + self.index, self.index +
                                  [os.path.getsize(path)])
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # TODO: support slices
        if not isinstance(idx, int):
            return TypeError('index must be integer')
        self.file.seek(self.index[idx])
        return self.file.read(self.lengths[idx])
