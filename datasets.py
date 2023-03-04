#!python3

import os
import json
import glob
import tqdm
import argparse

import hashlib
from zipfile import ZipFile
import gdown


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']


def sha512_hash(filename):
    hasher = hashlib.sha512()
    with open(filename, 'rb') as fp:
        while True:
            content = fp.read(10 * 1024 * 1024)
            if not content:
                break
            hasher.update(content)
    return hasher.hexdigest()


def download_video(ids):
    print('download and verify original video files of:', ' '.join(ids))
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100'))
    with open(os.path.join(basedir, 'videos.json'), 'r') as fp:
        videos = json.load(fp)
    videos = {v['id']: v for v in videos}
    basedir = os.path.join(basedir, 'videos')
    if not os.access(basedir, os.W_OK):
        os.mkdir(basedir)
    for video_id in ids:
        assert len(videos[video_id]['gid']) == 33, videos[video_id]
        filename = os.path.join(basedir, videos[video_id]['filename'])
        print('download', videos[video_id]['gid'], '=>', filename)
        gdown.download(id=videos[video_id]['gid'], output=filename, use_cookies=False)
        print('verify SHA512 of', filename, end=' ... ', flush=True)
        checksum = sha512_hash(filename)
        assert checksum.lower() == videos[video_id]['file']['sha512'].lower(), 'SHA512 not matching, file corrupted'
        print('passed')


def download_image(ids):
    print('download and verify pre-extracted training images LMDB of:', ' '.join(ids))
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100'))
    with open(os.path.join(basedir, 'images.json'), 'r') as fp:
        images = json.load(fp)
    images = {im['id']: im for im in images}
    basedir = os.path.join(basedir, 'images')
    if not os.access(basedir, os.W_OK):
        os.mkdir(basedir)
    for video_id in ids:
        assert len(images[video_id]['gid']) == 33, images[video_id]
        foldername = os.path.join(basedir, video_id)
        if not os.access(foldername, os.W_OK):
            os.mkdir(foldername)
        print('download', images[video_id]['gid'], '=>', foldername)
        gdown.download_folder(id=images[video_id]['gid'], output=foldername, use_cookies=False)
        with open(os.path.join(foldername, 'frames.json'), 'r') as fp:
            frames_meta = json.load(fp)
        assert frames_meta['meta']['id'] == video_id, 'wrong video ID: ' + frames_meta['meta']['id']
        for lmdb_filename in ['data.mdb', 'lock.mdb']:
            print('verify SHA512 of', lmdb_filename, end=' ... ', flush=True)
            checksum = sha512_hash(os.path.join(foldername, lmdb_filename))
            assert checksum.lower() == frames_meta['hash'][lmdb_filename]['sha512'].lower(), 'SHA512 not matching, file corrupted'
            print('passed')


def extract_image(ids):
    from scenes100.training_frames import TrainingFrames

    print('extracted training image files from LMDB of:', ' '.join(ids))
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100', 'images'))
    for video_id in ids:
        print('extract from:', os.path.join(basedir, video_id))
        for filename in ['data.mdb', 'lock.mdb', 'frames.json']:
            assert os.access(os.path.join(basedir, video_id, filename), os.R_OK), 'file not readable: ' + filename
        dst = TrainingFrames(video_id)
        print(dst)
        dst._extract()


def download_mscoco():
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco'))
    with open(os.path.join(basedir, 'mscoco.json'), 'r') as fp:
        files = json.load(fp)
    for f in files:
        filename = os.path.join(basedir, f['filename'])
        print('download', f['gid'], '=>', filename)
        gdown.download(id=f['gid'], output=filename, use_cookies=False)
        print('verify SHA512 of', filename, end=' ... ', flush=True)
        checksum = sha512_hash(filename)
        assert checksum.lower() == f['sha512'].lower(), 'SHA512 not matching, file corrupted'
        print('passed')


def extract_mscoco():
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco'))
    with open(os.path.join(basedir, 'mscoco.json'), 'r') as fp:
        files = json.load(fp)
    # for f in files:
    #     assert os.access(os.path.join(basedir, f['filename']), os.R_OK), 'file not readable: ' + f['filename']
    for prefix in ['images', 'inpaint_mask']:
        if not os.access(os.path.join(basedir, prefix), os.W_OK):
            os.mkdir(os.path.join(basedir, prefix))
        for postfix in ['val2017', 'train2017']:
            zipfilename = os.path.join(basedir, prefix + '_' + postfix + '.zip')
            foldername = os.path.join(basedir, prefix, postfix)
            print('extract', zipfilename, '=>', foldername)
            if not os.access(foldername, os.W_OK):
                os.mkdir(foldername)
            with ZipFile(zipfilename, 'r') as zfp:
                zfp.extractall(path=foldername)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download or extract datasets')
    parser.add_argument('--opt', type=str, default='', choices=['', 'download', 'extract'], help='operation to perform')
    parser.add_argument('--target', type=str, default='video', choices=['video', 'image', 'annotation', 'pseudolabel', 'mscoco'], help='target of operation')
    parser.add_argument('--ids', type=str, nargs='+', default=['001'], choices=['all'] + video_id_list, help='video IDs of operation, <all> means all 100 videos')
    args = parser.parse_args()

    args.ids = list(sorted(list(set(args.ids))))
    if 'all' in args.ids:
        assert len(args.ids) == 1, 'when <all> video IDs are specified, individual video IDs cannot be included'
        args.ids = video_id_list

    if args.opt == '':
        pass
    elif args.opt == 'download':
        if args.target == 'video':
            download_video(args.ids)
        elif args.target == 'image':
            download_image(args.ids)
        elif args.target == 'annotation':
            download_annotation()
        elif args.target == 'pseudolabel':
            download_pseudo_label()
        elif args.target == 'mscoco':
            download_mscoco()
    elif args.opt == 'extract':
        if args.target == 'video':
            print('cannot <extract> videos')
        elif args.target == 'image':
            extract_image(args.ids)
        elif args.target == 'annotation':
            extract_annotation()
        elif args.target == 'pseudolabel':
            extract_pseudo_label()
        elif args.target == 'mscoco':
            extract_mscoco()
