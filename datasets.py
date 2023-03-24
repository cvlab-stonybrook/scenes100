#!python3

import os
import json
import sys
import argparse
import subprocess

import hashlib
from zipfile import ZipFile

from adaptation.constants import video_id_list

baseurl = 'https://vision.cs.stonybrook.edu/~zekun/scenes100/'


def curl_download(url, filename):
    curl = str(subprocess.run(['which', 'curl'], capture_output=True, text=True, env=os.environ).stdout).strip()
    cmd = [curl, '--insecure', '-C', '-', url, '--output', filename]
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr).communicate()


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
        url = baseurl + 'videos/' + videos[video_id]['filename']
        filename = os.path.join(basedir, videos[video_id]['filename'])
        print('download', url, '=>', filename)
        curl_download(url, filename)
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
        foldername = os.path.join(basedir, video_id)
        if not os.access(foldername, os.W_OK):
            os.mkdir(foldername)
        for f in ['frames.json', 'lock.mdb', 'data.mdb']:
            url = baseurl + 'train_lmdb/' + video_id + '_' + f
            filename = os.path.join(foldername, f)
            print('download', url, '=>', filename)
            curl_download(url, filename)
        with open(os.path.join(foldername, 'frames.json'), 'r') as fp:
            frames_meta = json.load(fp)
        assert frames_meta['meta']['id'] == video_id, 'wrong video ID: ' + frames_meta['meta']['id']
        for f in ['data.mdb', 'lock.mdb']:
            print('verify SHA512 of', f, end=' ... ', flush=True)
            checksum = sha512_hash(os.path.join(foldername, f))
            assert checksum.lower() == frames_meta['hash'][f]['sha512'].lower(), 'SHA512 not matching, file corrupted'
            print('passed')


def extract_image(ids):
    from scenes100.training_frames import TrainingFrames
    print('extract training image files from LMDB of:', ' '.join(ids))
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100', 'images'))
    for video_id in ids:
        print('extract from:', os.path.join(basedir, video_id))
        for filename in ['data.mdb', 'lock.mdb', 'frames.json']:
            assert os.access(os.path.join(basedir, video_id, filename), os.R_OK), 'file not readable: ' + filename
        dst = TrainingFrames(video_id)
        print(dst)
        dst._extract()


def _decode(filename, foldername, ifilelist, idxlist):
    import imageio
    import skvideo.io
    import tqdm
    import glob
    print('decode %s\n%d frames to %s' % (filename, len(ifilelist), foldername))
    reader = skvideo.io.vreader(filename)
    for i in tqdm.tqdm(range(0, max(idxlist) + 5), ascii=True):
        try:
            frame = next(reader)
        except StopIteration:
            break
        if i not in idxlist:
            continue
        fn = os.path.join(foldername, ifilelist[idxlist.index(i)])
        jpeg_bytes = imageio.v2.imwrite('<bytes>', frame, plugin='pillow', format='JPEG', quality=80)
        with open(fn, 'wb') as fp:
            fp.write(jpeg_bytes)
    print('%d JPEG files saved' % len(glob.glob(os.path.join(foldername, '*.jpg'))))
    reader.close()


def decode_image(ids):
    print('decode training image files from original videos of:', ' '.join(ids))
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100'))
    with open(os.path.join(basedir, 'videos.json'), 'r') as fp:
        videos = json.load(fp)
    videos = {v['id']: v for v in videos}
    basedir = os.path.join(basedir, )
    for video_id in ids:
        filename = os.path.join(basedir, 'videos', videos[video_id]['filename'])
        print('original video file:', filename)
        assert os.access(filename, os.R_OK), 'video file not readable'
        checksum = sha512_hash(filename)
        assert checksum.lower() == videos[video_id]['file']['sha512'].lower(), 'SHA512 not matching, file corrupted'

        foldername = os.path.join(basedir, 'images', video_id)
        if not os.access(foldername, os.W_OK):
            os.mkdir(foldername)
        url = baseurl + 'train_lmdb/' + video_id + '_frames.json'
        filename_meta = os.path.join(foldername, 'frames.json')
        print('download', url, '=>', filename_meta)
        curl_download(url, filename_meta)
        with open(os.path.join(foldername, 'frames.json'), 'r') as fp:
            frames_meta = json.load(fp)
        assert frames_meta['meta']['id'] == video_id, 'wrong video ID: ' + frames_meta['meta']['id']
        ifilelist = frames_meta['ifilelist']
        idxlist = list(map(lambda f: int(f[:f.find('.')]), ifilelist))

        foldername = os.path.join(basedir, 'images', video_id, 'jpegs')
        if not os.access(foldername, os.W_OK):
            os.mkdir(foldername)
        _decode(filename, foldername, ifilelist, idxlist)


def download_mscoco():
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco'))
    with open(os.path.join(basedir, 'mscoco.json'), 'r') as fp:
        files = json.load(fp)
    if not os.access(os.path.join(basedir, 'models'), os.W_OK):
        os.mkdir(os.path.join(basedir, 'models'))
    for f in files['checkpoint']:
        url = baseurl + 'mscoco/' + f['filename']
        filename = os.path.join(basedir, 'models', f['filename'])
        print('download', url, '=>', filename)
        curl_download(url, filename)
        print('verify SHA512 of', filename, end=' ... ', flush=True)
        checksum = sha512_hash(filename)
        assert checksum.lower() == f['sha512'].lower(), 'SHA512 not matching, file corrupted'
        print('passed')
    for f in files['dataset']:
        url = baseurl + 'mscoco/' + f['filename']
        filename = os.path.join(basedir, f['filename'])
        print('download', url, '=>', filename)
        curl_download(url, filename)
        print('verify SHA512 of', filename, end=' ... ', flush=True)
        checksum = sha512_hash(filename)
        assert checksum.lower() == f['sha512'].lower(), 'SHA512 not matching, file corrupted'
        print('passed')


def extract_mscoco():
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco'))
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


def download_annotation():
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100'))
    with open(os.path.join(basedir, 'annotations.json'), 'r') as fp:
        files = json.load(fp)
    for f in files:
        url = baseurl + 'annotation/' + f['filename']
        filename = os.path.join(basedir, f['filename'])
        print('download', url, '=>', filename)
        curl_download(url, filename)
        print('verify SHA512 of', filename, end=' ... ', flush=True)
        checksum = sha512_hash(filename)
        assert checksum.lower() == f['sha512'].lower(), 'SHA512 not matching, file corrupted'
        print('passed')


def extract_annotation():
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100'))
    for zipfilename, foldername in [('manual_valid.zip', 'annotation'), ('background_train.zip', 'train_background'), ('background_valid.zip', 'valid_background'), ('pseudo_label_r101_track.zip', 'train_pseudo_label'), ('pseudo_label_r50_track.zip', 'train_pseudo_label'), ('pseudo_label_r101.zip', 'train_pseudo_label'), ('pseudo_label_r50.zip', 'train_pseudo_label')]:
        foldername = os.path.join(basedir, foldername)
        zipfilename = os.path.join(basedir, zipfilename)
        print('extract', zipfilename, '=>', foldername)
        if not os.access(foldername, os.W_OK):
            os.mkdir(foldername)
        with ZipFile(zipfilename, 'r') as zfp:
            zfp.extractall(path=foldername)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download or extract datasets')
    parser.add_argument('--opt', type=str, default='', choices=['', 'download', 'extract', 'decode'], help='operation to perform')
    parser.add_argument('--target', type=str, default='video', choices=['video', 'image', 'annotation', 'mscoco'], help='target of operation')
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
        elif args.target == 'mscoco':
            download_mscoco()
    elif args.opt == 'extract':
        if args.target == 'video':
            print('cannot <extract> videos')
        elif args.target == 'image':
            extract_image(args.ids)
        elif args.target == 'annotation':
            extract_annotation()
        elif args.target == 'mscoco':
            extract_mscoco()
    elif args.opt == 'decode':
        assert args.target == 'image', 'can only decode images'
        decode_image(args.ids)
