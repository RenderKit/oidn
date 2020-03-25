## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import struct
import numpy as np

# Tensor Archive (TZA) file format
VERSION = (2, 0)
_MAGIC = 0x41D7

# Writes tensors to a TZA file
class Writer(object):
  # Creates a new file
  def __init__(self, filename):
    self._table = []
    self._file = open(filename, 'wb')
    self._write_header()

  def __enter__(self):
    return self

  def __exit__(self, type, value, tb):
    self.close()

  # Encodes data type
  def _encode_dtype(self, dtype):
    if dtype == np.float32:
      return 'f'
    elif dtype == np.float16:
      return 'h'
    else:
      raise ValueError('unsupported tensor data type')
  
  def _write_uint8(self, x):
    self._file.write(struct.pack('B', x))

  def _write_uint16(self, x):
    self._file.write(struct.pack('H', x))

  def _write_uint32(self, x):
    self._file.write(struct.pack('I', x))

  def _write_uint64(self, x):
    self._file.write(struct.pack('Q', x))

  # Writes a raw byte string (without length) to the file
  def _write_raw_str(self, str):
    self._file.write(str.encode(encoding='ascii'))

  # Writes an UTF-8 string to the file
  def _write_str(self, str):
    data = str.encode()
    self._write_uint16(len(data))
    self._file.write(data)

  # Writes padding to the file
  def _write_pad(self, alignment=64):
    offset = self._file.tell()
    pad = (offset + alignment - 1) // alignment * alignment - offset
    for _ in range(pad):
      self._write_uint8(0)

  # Writes the header to the file
  def _write_header(self):
    self._write_uint16(_MAGIC)
    self._write_uint8(VERSION[0])
    self._write_uint8(VERSION[1])
    self._write_uint64(0) # placeholder for the table offset

  # Writes the table to the file
  def _write_table(self):
    self._write_pad()
    table_offset = self._file.tell()
    self._write_uint32(len(self._table))

    for name, shape, layout, dtype, offset in self._table:
      self._write_str(name)
      ndims = len(shape)
      self._write_uint8(ndims)
      for dim in shape:
        self._write_uint32(dim)
      self._write_raw_str(layout)
      self._write_raw_str(dtype)
      self._write_uint64(offset)

    self._file.seek(4) # skip magic and version
    self._write_uint64(table_offset)

  # Writes a tensor to the file
  def write(self, name, tensor, layout):
    shape = tensor.shape
    ndims = len(shape)
    if len(layout) != ndims:
      raise ValueError('invalid tensor layout')
    dtype = self._encode_dtype(tensor.dtype)
    self._write_pad()
    offset = self._file.tell()

    self._table.append((name, shape, layout, dtype, offset))
    tensor.tofile(self._file)

  # Closes the file
  def close(self):
    self._write_table()
    self._file.close()

# Reads tensors from a TZA file
class Reader(object):
  # Opens a file
  def __init__(self, filename):
    self.tensors = {}
    self.layouts = {}
    self._file = open(filename, 'rb')
    self._read_header()
    self._read_table()
    self._map_tensors()
    del self._table

  def __enter__(self):
    return self

  def __exit__(self, type, value, tb):
    self.close()

  def __iter__(self):
    return iter(self.tensors)

  def __len__(self):
    return len(self.tensors)

  def __getitem__(self, key):
    return self.tensors[key]

  # Decodes data type
  def _decode_dtype(self, dtype):
    if dtype == 'f':
      return np.float32
    elif dtype == 'h':
      return np.float16
    else:
      raise ValueError('unsupported tensor data type')

  def _read_uint8(self):
    return struct.unpack('B', self._file.read(1))[0]

  def _read_uint16(self):
    return struct.unpack('H', self._file.read(2))[0]

  def _read_uint32(self):
    return struct.unpack('I', self._file.read(4))[0]

  def _read_uint64(self):
    return struct.unpack('Q', self._file.read(8))[0]

  # Reads a raw byte string (without length) from the file
  def _read_raw_str(self, size):
    return self._file.read(size).decode(encoding='ascii')

  # Reads an UTF-8 string from the file
  def _read_str(self):
    n = self._read_uint16()
    data = self._file.read(n)
    return data.decode()

  # Reads the header from the file
  def _read_header(self):
    magic = self._read_uint16()
    if magic != _MAGIC:
      raise ValueError('invalid tensor format')
    self._version = (self._read_uint8(), self._read_uint8())
    if self._version[0] != VERSION[0]:
      raise ValueError('unsupported tensor format version')
    self._table_offset = self._read_uint64()

  # Reads the table from the file
  def _read_table(self):
    self._table = []
    self._file.seek(self._table_offset)
    num_tensors = self._read_uint32()

    for _ in range(num_tensors):
      name = self._read_str()
      ndims = self._read_uint8()
      shape = tuple(self._read_uint32() for _ in range(ndims))
      layout = self._read_raw_str(ndims)
      dtype = self._read_raw_str(1)
      offset = self._read_uint64()

      self._table.append((name, shape, layout, dtype, offset))

  # Maps the tensors into memory
  def _map_tensors(self):
    # Get the size of the file
    self._file.seek(0, 2) # seek to the end
    buffer_size = self._file.tell()

    # Map the entire file into memory
    buffer = np.memmap(self._file,
                       dtype=np.uint8,
                       mode='r',
                       shape=(buffer_size))

    # Add the tensors
    for name, shape, layout, dtype, offset in self._table:
      self.tensors[name] = np.ndarray(shape,
                                      dtype=self._decode_dtype(dtype),
                                      buffer=buffer,
                                      offset=offset)
      self.layouts[name] = layout
 
  # Closes the file
  def close(self):
    self._file.close()
