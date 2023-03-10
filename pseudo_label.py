#!python3

import os
import sys
import time
import json
import gzip
import random
import tqdm
import imageio
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))
from adaptation.constants import video_id_list, thing_classes


def detect():
    import detectron2
    from detectron2.engine import DefaultPredictor
    from adaptation.base_model_cfg import get_cfg_base_model
    
    imagedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100', 'images', args.id))
    with open(os.path.join(imagedir, 'frames.json'), 'r') as fp:
        ifilelist = json.load(fp)['ifilelist']

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    predictor = DefaultPredictor(cfg)
    print('detect objects with %s in video %s %s' % (args.model, args.id, imagedir), flush=True)

    frame_objs = []
    for f in tqdm.tqdm(ifilelist, ascii=True):
        im = detectron2.data.detection_utils.read_image(os.path.join(imagedir, 'jpegs', f), format='BGR')
        instances = predictor(im)['instances'].to('cpu')
        frame_objs.append({
            # bbox has format [x1, y1, x2, y2]
            'bbox': instances.pred_boxes.tensor.numpy().tolist(),
            'score': instances.scores.numpy().tolist(),
            'label': instances.pred_classes.numpy().tolist()
        })

    result_json_zip = os.path.join(args.outputdir, '%s_detect_%s.json.gz' % (args.id, args.model))
    with gzip.open(result_json_zip, 'wt') as fp:
        fp.write(json.dumps({'model': args.model, 'classes': thing_classes, 'frames': ifilelist, 'dets': frame_objs, 'args': vars(args)}))
    print('results saved to:', result_json_zip)


def track_video_multi():
    assert os.path.isdir(args.pytracking_dir), args.pytracking_dir + ' is not a directory'
    assert os.path.isdir(args.cuda_dir), args.cuda_dir + ' is not a directory'
    assert os.access(args.detect_file, os.R_OK), 'detection file not readable: ' + args.detect_file
    
    os.environ['CUDA_HOME'] = args.cuda_dir
    sys.path.append(args.pytracking_dir)
    from pytracking.evaluation import Tracker as TrackerWrapper

    def run_tracker_multi(vid_arr, dets):
        track_bboxes = [[] for _ in range(0, len(dets))]
        bbox, scores, labels = dets[0]['bbox'], dets[0]['score'], dets[0]['label']
        _n, _t = 0, time.time()
        for j in range(0, len(scores)):
            if scores[j] < args.sot_score_thres:
                continue
            x1, y1, x2, y2 = bbox[j]
            if x2 - x1 < args.sot_min_bbox or y2 - y1 < args.sot_min_bbox:
                continue
            area_0 = (x2 - x1) * (y2 - y1)

            track_bboxes[0].append({'class': labels[j], 'bbox': [x1, y1, x2, y2], 'init_score': scores[j], 'track_length': 0})
            tracker.initialize(vid_arr[0], {'init_bbox': [x1, y1, x2 - x1, y2 - y1], 'init_object_ids': [1], 'object_ids': [1], 'sequence_object_ids': [1]})
            for i_track in range(1, len(dets)):
                out = tracker.track(vid_arr[i_track])
                _n += 1
                x, y, w, h = out['target_bbox']
                if x < 0:
                    break
                area_i = w * h
                if area_i > area_0 * 2 or area_i < area_0 / 2:
                    break
                else:
                    track_bboxes[i_track].append({'class': labels[j], 'bbox': [x, y, x + w, y + h], 'init_score': scores[j], 'track_length': i_track})

        return track_bboxes, _n, time.time() - _t

    imagedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100', 'images', args.id))
    with open(os.path.join(imagedir, 'frames.json'), 'r') as fp:
        fps = json.load(fp)['sample_fps']
    with gzip.open(args.detect_file, 'rt') as fp:
        data = json.loads(fp.read())
    ifilelist, dets = data['frames'], data['dets']
    assert len(dets) == len(ifilelist), 'detection results and dataset mis-match'
    print('track objects in video %s %s, initialized from %s, %d frames FPS=%.1f' % (args.id, imagedir, args.detect_file, len(ifilelist), fps), flush=True)

    random.seed(42)
    if args.sot_skip < 0:
        sot_every_frame = 1
    else:
        sot_every_frame = int(fps * args.sot_skip)
    sot_num_frames = int(fps * args.sot_max_length)
    sot_idx_list = np.arange(0, len(ifilelist), sot_every_frame)[2 : -(sot_num_frames + 1)].tolist()
    print('start tracking every %d frames, total %d init points, maximum track %d frames, minimum init detection score %.3f, minimum init bbox width %d' % (sot_every_frame, len(sot_idx_list), sot_num_frames, args.sot_score_thres, args.sot_min_bbox))

    wrapper = TrackerWrapper('dimp', 'dimp50')
    params = wrapper.get_parameters()
    params.debug = 0
    params.tracker_name = wrapper.name
    params.param_name = wrapper.parameter_name
    params.output_not_found_box = True
    tracker = wrapper.create_tracker(params)
    tracker.initialize_features()
    print('DiMP-50 tracker initialized', flush=True)

    tracked_total, time_total = 0, 0
    boxes_forward, boxes_backward = [[] for _ in range(0, len(dets))], [[] for _ in range(0, len(dets))]
    sot_json_zip = os.path.join(args.outputdir, os.path.basename(args.detect_file)[:-8] + '_DiMP.json.gz')
    for init_i in tqdm.tqdm(sot_idx_list, ascii=True):
        vid_i = [os.path.join(imagedir, 'jpegs', f) for f in ifilelist[init_i : init_i + sot_num_frames]]
        vid_i = np.stack([np.asarray(imageio.imread(f)) for f in vid_i], axis=0)
        dets_i = dets[init_i : init_i + sot_num_frames]

        forward_i, _n1, _t1 = run_tracker_multi(vid_i, dets_i)
        backward_i, _n2, _t2 = run_tracker_multi(vid_i[::-1], dets_i[::-1])
        backward_i = backward_i[::-1]
        tracked_total += _n1 + _n2
        time_total += _t1 + _t2
        assert len(forward_i) == len(backward_i) and len(forward_i) == len(dets_i)
        for i in range(init_i, init_i + sot_num_frames):
            boxes_forward[i] = boxes_forward[i] + forward_i[i - init_i]
            boxes_backward[i] = boxes_backward[i] + backward_i[i - init_i]

        if 0 == sot_idx_list.index(init_i) % 100:
            with gzip.open(sot_json_zip, 'wt') as fp:
                fp.write(json.dumps({'forward': boxes_forward, 'backward': boxes_backward, 'args': vars(args)}))
    with gzip.open(sot_json_zip, 'wt') as fp:
        fp.write(json.dumps({'forward': boxes_forward, 'backward': boxes_backward, 'args': vars(args)}))
    print('results saved to:', sot_json_zip)
    print('finished tracking %d frames in %.1f seconds (%.3f frames/second)' % (tracked_total, time_total, tracked_total / time_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pseudo Labeling Script')
    parser.add_argument('--opt', type=str, choices=['detect', 'track'], help='option')
    parser.add_argument('--id', type=str, choices=video_id_list, help='video ID')
    parser.add_argument('--model', type=str, choices=['r50-fpn-3x', 'r101-fpn-3x'], help='detection model')
    parser.add_argument('--ckpt', type=str, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.', help='save detection and tracking results to this directory')

    parser.add_argument('--pytracking_dir', type=str, help='root directory of PyTracking repository')
    parser.add_argument('--cuda_dir', type=str, help='root directory of CUDA toolkit')
    parser.add_argument('--detect_file', type=str, help='file that contains detected pseudo bounding boxes')
    parser.add_argument('--sot_score_thres', type=float, default=0.9, help='minimum detection score to start tracking')
    parser.add_argument('--sot_min_bbox', type=int, default=50, help='minimum detection box size to start tracking')
    parser.add_argument('--sot_skip', type=float, default=-1, help='interval of video segments for tracking in seconds')
    parser.add_argument('--sot_max_length', type=float, default=2, help='maximum seconds of tracks')
    args = parser.parse_args()
    print(args)

    if args.opt == 'detect':
        detect()
    elif args.opt == 'track':
        track_video_multi()


'''
python pseudo_label.py --opt detect --id 001 --model r50-fpn-3x
python pseudo_label.py --opt track --id 001 --model r101-fpn-3x --sot_skip 5 --sot_max_length 2
'''
