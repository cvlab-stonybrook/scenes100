#!python3

import os
import glob
import json
import tqdm
import skimage.io
import skvideo.io
import hashlib
import numpy as np
import imageio
import lmdb
import collections.abc
from multiprocessing import Pool as ProcessPool
import torch.utils.data as torchdata


def decode_frames(meta):
    def sha_hashes(vfilename):
        results = {}
        for desc, hasher in [('sha1', hashlib.sha1()), ('sha512', hashlib.sha512())]:
            with open(vfilename, 'rb') as fp:
                while True:
                    content = fp.read(10 * 1024 * 1024)
                    if not content:
                        break
                    hasher.update(content)
            results[desc] = hasher.hexdigest()
        return results

    vfilename = os.path.join(os.path.dirname(__file__), '..', 'videos', meta['filename'])
    video_id = meta['id']
    if not os.access(vfilename, os.R_OK):
        print('cannot read file', vfilename)
        return

    fps, F, H, W = meta['video']['fps'], meta['video']['frames'] - 1, meta['video']['H'], meta['video']['W']
    sample_fps, encoding = 5, 'utf-8'
    framesdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'train_lmdb', video_id)

    annodir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotations')
    valid_list = glob.glob(os.path.join(annodir, 'images', video_id + '_*.jpg'))
    weaksupervised_list = glob.glob(os.path.join(annodir, 'sparse', video_id + '_*.jpg'))
    valid_list = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in valid_list]
    weaksupervised_list = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in weaksupervised_list]
    assert len(valid_list) > 0 and len(weaksupervised_list) > 0, '%s %s %s' % (video_id, valid_list, weaksupervised_list)
    valid_list = {i : list(filter(lambda x: x >= 0 and x < F, range(int(i - fps * 2), int(i + fps * 2)))) for i in valid_list}
    weaksupervised_list = {i : list(filter(lambda x: x >= 0 and x < F, range(int(i - fps * 2), int(i + fps * 2)))) for i in weaksupervised_list}
    valid_context_set = []
    for i in valid_list:
        valid_context_set = valid_context_set + valid_list[i]
    valid_context_set = set(valid_context_set)
    weaksupervised_context_set = []
    for i in weaksupervised_list:
        weaksupervised_context_set = weaksupervised_context_set + weaksupervised_list[i]
    weaksupervised_context_set = set(weaksupervised_context_set)

    output_json = os.path.join(framesdir, 'frames.json')
    if os.access(output_json, os.R_OK):
        print('%s exists, skipped' % output_json)
        return

    if not os.access(framesdir, os.W_OK):
        os.mkdir(framesdir)
    os.mkdir(os.path.join(framesdir, 'weaksupervised_context'))
    os.mkdir(os.path.join(framesdir, 'valid_context'))
    lmdb_map_size_inc = 64 * 1024 * 1024
    lmdb_map_size = lmdb_map_size_inc
    saved_bytes = 0
    env = lmdb.open(framesdir, map_size=lmdb_map_size, metasync=True, sync=True)
    txn = env.begin(write=True)

    desc = '%s... %04dx%04d %.1f fps F=%d' % (meta['filename'][:15], H, W, fps, F)
    train_frame_count = int(fps * 1.5 * 3600)
    train_frame_idx = set(np.arange(0, train_frame_count, fps / sample_fps).astype(np.int32).tolist())
    ifilelist = []
    reader = skvideo.io.vreader(vfilename)
    for i in tqdm.tqdm(range(0, F), ascii=True, desc=desc):
        try:
            frame = next(reader)
        except StopIteration:
            break
        fn = '%08d.jpg' % i
        if i in train_frame_idx:
            jpeg_bytes = imageio.imwrite('<bytes>', frame, plugin='pillow', format='JPEG', quality=80)
            fn_bytes = fn.encode(encoding)
            saved_bytes += len(jpeg_bytes) + len(fn_bytes)
            if saved_bytes > lmdb_map_size * 0.95:
                txn.commit()
                env.close()
                lmdb_map_size += lmdb_map_size_inc
                env = lmdb.open(framesdir, map_size=lmdb_map_size, metasync=True, sync=True)
                txn = env.begin(write=True)
            ret = txn.put(fn_bytes, jpeg_bytes)
            assert ret, 'put data failed'
            ifilelist.append(fn)
        if i in valid_context_set:
            skimage.io.imsave(os.path.join(framesdir, 'valid_context', fn), frame, quality=80)
        if i in weaksupervised_context_set:
            skimage.io.imsave(os.path.join(framesdir, 'weaksupervised_context', fn), frame, quality=80)
    reader.close()
    txn.commit()
    env.close()

    hashes = {}
    for f in ['data.mdb', 'lock.mdb']:
        hashes[f] = sha_hashes(os.path.join(framesdir, f))
    with open(output_json, 'w') as fp:
        json.dump({'map_size': lmdb_map_size, 'encoding': encoding, 'ifilelist': ifilelist, 'valid_list': valid_list, 'weaksupervised_list': weaksupervised_list, 'sample_fps': sample_fps, 'meta': meta, 'hash': hashes}, fp)


class TrainingFrames(collections.abc.Sequence):
    def __init__(self, video_id):
        self.video_id = video_id
        self.lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'images', 'train_lmdb', video_id))
        assert type(self.video_id) == type('000') and len(self.video_id) == 3, '<video_id> should be the 3-digits video ID string'
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
        os.mkdir(os.path.join(self.lmdb_path, 'jpegs'))
        for i in tqdm.tqdm(range(0, len(self.ifilelist)), ascii=True, desc='extracting & saving JPEG images'):
            fn = self.ifilelist[i]
            jpeg_bytes = self.lmdb_txn.get(fn.encode(self.meta['encoding']))
            with open(os.path.join(self.lmdb_path, 'jpegs', fn), 'wb') as fp:
                fp.write(jpeg_bytes)


class TrainingFramesDataset(torchdata.Dataset):
    def __init__(self, video_id):
        super(TrainingFramesDataset, self).__init__()
        self.dst = TrainingFrames(video_id)
    def __len__(self):
        return len(self.dst)
    def __getitem__(self, i):
        return self.dst[i]


if __name__ == '__main__':
    # dst = TrainingFrames('003')
    # print(dst)
    # dst = dst[:25]
    # for _ in tqdm.tqdm(dst, ascii=True): pass
    # reader = iter(dst)
    # while True:
    #     try: im, frame_id, fn, i = next(reader)
    #     except StopIteration: break
    #     print(i, frame_id, fn, im.dtype, im.shape)
    # print('')
    # dst = dst[::-1]
    # for im, frame_id, fn, i in dst: print(i, frame_id, fn, im.dtype, im.shape)
    # print('')
    # dst = dst[5:20]
    # for im, frame_id, fn, i in dst: print(i, frame_id, fn, im.dtype, im.shape)
    # print('')
    # dst = dst[0:-1:3]
    # for im, frame_id, fn, i in dst: print(i, frame_id, fn, im.dtype, im.shape)

    # loader = torchdata.DataLoader(TrainingFramesDataset('172'), batch_size=5, shuffle=False, num_workers=2)
    # for i, (im, frame_id, fn, idx) in enumerate(loader):
    #     print(idx, frame_id, fn, im.size(), im.dtype)
    #     if i > 4:
    #         break

    # import zlib
    # import matplotlib.pyplot as plt
    # quality_bytes = {Q: 0 for Q in range(60, 101)}
    # quality_bytes_z = {Q: 0 for Q in quality_bytes}
    # reader = skvideo.io.vreader(os.path.join('..', 'videos', '001.JacksonHoleTownSquare_20200910_210625.mp4'))
    # for i in tqdm.tqdm(range(0, 100), ascii=True):
    #     frame = next(reader)
    #     if 0 != i % 10: continue
    #     for Q in quality_bytes:
    #         jpeg_bytes = imageio.imwrite('<bytes>', frame, plugin='pillow', format='JPEG', quality=Q)
    #         z_bytes = zlib.compress(jpeg_bytes, level=9)
    #         quality_bytes[Q] += len(jpeg_bytes)
    #         quality_bytes_z[Q] += len(z_bytes)
    # reader.close()
    # plt.figure()
    # plt.plot(list(quality_bytes.keys()), list(quality_bytes.values()))
    # plt.plot(list(quality_bytes.keys()), list(quality_bytes_z.values()))
    # plt.ylim(0, 1.05 * max(list(quality_bytes.values())))
    # plt.show()
    # exit(0)

    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    all_ids = list(map(lambda x: x['id'], files))

    import argparse
    parser = argparse.ArgumentParser(description='Decoding & Extraction')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--ids', nargs='+', default=[], choices=all_ids, help='video IDs')
    parser.add_argument('--procs', type=int, help='decoding processes')
    args = parser.parse_args()
    print(args)

    if args.opt == 'decode':
        assert len(args.ids) > 0 and args.procs > 0
        files = list(filter(lambda f: not os.access(os.path.join(os.path.dirname(__file__), '..', 'images', 'train_lmdb', f['id'], 'frames.json'), os.R_OK), files))
        files = list(filter(lambda f: f['id'] in args.ids, files))
        print('decode %s with %d processes' % (list(map(lambda f: f['id'], files)), args.procs))
        pool = ProcessPool(processes=args.procs)
        _ = pool.map_async(decode_frames, files).get()
        pool.close()
        pool.join()

    elif args.opt == 'ext':
        assert len(args.ids) > 0
        for i in args.ids:
            dst = TrainingFrames(i)
            print(dst)
            dst._extract()

    else: pass
