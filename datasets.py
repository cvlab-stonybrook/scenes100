#!python3

import os
import json
import glob
import tqdm
import argparse

import hashlib
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
    print('download and verify original video files of videos:', ' '.join(ids))
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
        print('verify SHA512 of', filename, end=' ... ')
        assert os.access(filename, os.R_OK), 'video file not readable: ' + filename
        checksum = sha512_hash(filename)
        assert checksum.lower() == videos[video_id]['file']['sha512'].lower(), 'SHA512 not matching, file corrupted'
        print('passed')


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
            download_annotation(args.ids)
        elif args.target == 'pseudolabel':
            download_pseudo_label(args.ids)
        elif args.target == 'mscoco':
            download_mscoco(args.ids)
    elif args.opt == 'extract':
        if args.target == 'video':
            print('cannot <extract> videos')
        elif args.target == 'image':
            extract_image(args.ids)
        elif args.target == 'annotation':
            extract_annotation(args.ids)
        elif args.target == 'pseudolabel':
            extract_pseudo_label(args.ids)
        elif args.target == 'mscoco':
            extract_mscoco(args.ids)
