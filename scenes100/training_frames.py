#!python3

import os
import json
import tqdm
import collections.abc

import numpy as np
import imageio
import lmdb
import torch


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']


class TrainingFrames(collections.abc.Sequence):
    def __init__(self, video_id):
        self.video_id = video_id
        self.lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'images', video_id))
        assert self.video_id in video_id_list, '<video_id> should be one of: ' + ' '.join(video_id_list)
        assert os.access(os.path.join(self.lmdb_path, 'frames.json'), os.R_OK), 'frames.json not found in: ' + self.lmdb_path
        assert os.access(os.path.join(self.lmdb_path, 'data.mdb'), os.R_OK) and os.access(os.path.join(self.lmdb_path, 'lock.mdb'), os.R_OK), 'LMDB files not found in: ' + self.lmdb_path
        with open(os.path.join(self.lmdb_path, 'frames.json'), 'r') as fp:
            self.meta = json.load(fp)
        self.ifilelist = self.meta['ifilelist']
        self.lmdb_env, self.lmdb_txn = None, None
    def __del__(self):
        if not self.lmdb_env is None:
            self.lmdb_env.close()
    def __repr__(self):
        return 'TrainingFrames [video %s] [%dx%d] [%d frames] [%.1f fps] [%.2f GB]' % (self.video_id, self.meta['meta']['video']['H'], self.meta['meta']['video']['W'], len(self), self.meta['sample_fps'], os.path.getsize(os.path.join(self.lmdb_path, 'data.mdb')) / (1024 ** 3))

    def __len__(self):
        return len(self.ifilelist)
    def __getitem__(self, index):
        if isinstance(index, int):
            if self.lmdb_txn is None:
                self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
                self.lmdb_txn = self.lmdb_env.begin(write=False)
            fn, frame_id = self.ifilelist[index], int(self.ifilelist[index].split('.')[0])
            jpeg_bytes = self.lmdb_txn.get(fn.encode(self.meta['encoding']))
            im = imageio.imread(jpeg_bytes, format='JPEG')
            return np.array(im), frame_id, fn, index
        if isinstance(index, slice):
            sliced_dst = TrainingFrames(self.video_id)
            sliced_dst.ifilelist = self.ifilelist[index]
            return sliced_dst
        raise Exception('unsupported index: %s' % index)
    def __setitem__(self, index, item):
        raise NotImplementedError
    def __delitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        self.iter_index = -1
        return self
    def __next__(self):
        self.iter_index += 1
        if self.iter_index < len(self):
            return self[self.iter_index]
        else:
            raise StopIteration

    def __contains__(self, item):
        raise NotImplementedError
    def __reversed__(self):
        raise NotImplementedError
    def index(self, item):
        raise NotImplementedError
    def count(self, item):
        raise NotImplementedError

    def _extract(self):
        if self.lmdb_txn is None:
            self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
            self.lmdb_txn = self.lmdb_env.begin(write=False)
        foldername = os.path.join(self.lmdb_path, 'jpegs')
        assert not os.access(foldername, os.W_OK), foldername + ' already exists, cannot extract training frames'
        os.mkdir(foldername)
        for i in tqdm.tqdm(range(0, len(self.ifilelist)), ascii=True, desc='extracting & saving JPEG images'):
            fn = self.ifilelist[i]
            jpeg_bytes = self.lmdb_txn.get(fn.encode(self.meta['encoding']))
            with open(os.path.join(foldername, fn), 'wb') as fp:
                fp.write(jpeg_bytes)


class TrainingFramesDataset(torch.utils.data.Dataset):
    def __init__(self, video_id):
        super(TrainingFramesDataset, self).__init__()
        self.dst = TrainingFrames(video_id)
    def __len__(self):
        return len(self.dst)
    def __getitem__(self, i):
        return self.dst[i]


if __name__ == '__main__':
    pass
