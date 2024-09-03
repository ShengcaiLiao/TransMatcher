from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import time
import torch
import numpy as np
import scipy.io as sio

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('QAConv')
from reid import datasets
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint
from reid.evaluators import extract_features
from reid.loss.pairwise_matching_loss import PairwiseMatchingLoss

from transmatcher_match import TransMatcher

sys.path.append('../')
import restranmap as resmap


def get_test_data(dataname, data_dir, height, width, workers=8, test_batch=64):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, combine_all=False)

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
    ])

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=test_batch, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=test_batch, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, query_loader, gallery_loader


def pairwise_similarity(matcher, prob_fea, gal_fea, hei, wid, num_layers, gal_batch_size=4, prob_batch_size=4096):
    with torch.no_grad():
        num_gals = gal_fea.size(0)
        num_probs = prob_fea.size(0)
        score = torch.zeros(num_probs, num_gals)
        match_index = torch.zeros(num_probs, num_probs, num_layers, 4, hei, wid)
        matcher.eval()
        for i in range(0, num_probs, prob_batch_size):
            print('Compute similarity %d / %d...' % (i, num_probs), end='\r')
            j = min(i + prob_batch_size, num_probs)
            matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            for k in range(0, num_gals, gal_batch_size):
                k2 = min(k + gal_batch_size, num_gals)
                s, index = matcher(gal_fea[k: k2, :, :, :].cuda())
                score[i: j, k: k2] = s.cpu()
                match_index[i: j, k: k2, :, :, :, :] = index.cpu()
        # scale matching scores to make them visually more recognizable
        score = torch.sigmoid(score / 10)
    return score, match_index


def main(args):
    cudnn.benchmark = True

    # Create data loaders
    dataset, query_loader, gallery_loader = get_test_data(args.testset, args.data_dir, args.height, args.width, args.workers,
                                                          args.test_fea_batch)

    # Create model
    ibn_type = args.ibn
    if ibn_type == 'none':
        ibn_type = None
    model = resmap.create(args.arch, ibn_type=ibn_type, final_layer=args.final_layer, neck=args.neck, nhead=args.nhead, 
                num_encoder_layers=args.num_trans_layers - 1, dim_feedforward=args.dim_feedforward, dropout=args.dropout).cuda()
    num_features = model.num_features

    feamap_factor = {'layer2': 8, 'layer3': 16, 'layer4': 32}
    hei = args.height // feamap_factor[args.final_layer]
    wid = args.width // feamap_factor[args.final_layer]
    matcher = TransMatcher(hei * wid, num_features, args.num_trans_layers, args.dim_feedforward).cuda()

    for arg in sys.argv:
        print('%s ' % arg, end='')
    print('\n')

    # Criterion
    criterion = PairwiseMatchingLoss(matcher).cuda()

    print('Loading checkpoint...')
    checkpoint = load_checkpoint(osp.join(args.exp_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    criterion.load_state_dict(checkpoint['criterion'])

    model = nn.DataParallel(model).cuda()

    # Final test
    print('Evaluate the learned model:')
    t0 = time.time()

    feature0, _ = extract_features(model, query_loader, verbose=True)
    feature = torch.cat([feature0[f].unsqueeze(0) for f, _, _, _ in dataset.query], 0)
    del feature0

    # start_prob = 0
    # end_prob = 64
    # end_prob = min(feature.size(0), end_prob)
    # feature = feature[start_prob: end_prob]

    score, match_index = pairwise_similarity(matcher, feature, feature, hei, wid, 
                            args.num_trans_layers, args.test_gal_batch, args.test_prob_batch)

    del feature
    prob_score = match_index[:, :, :, 0, :, :].numpy()
    index_in_gal = match_index[:, :, :, 1, :, :].byte().numpy()
    del match_index

    score_embed = [matcher.decoder.layers[i].score_embed.view(hei, wid, hei, wid).detach().cpu() for i in range(args.num_trans_layers)]
    score_embed = torch.stack(score_embed, dim=0)
    test_prob_list = np.array([fname for fname, _, _, _ in dataset.query], dtype=np.object)
    test_prob_ids = [pid for _, pid, _, _ in dataset.query]
    test_prob_cams = [cam for _, _, cam, _ in dataset.query]
    test_score_file = osp.join(args.exp_dir, '%s_query_score.mat' % args.testset)
    sio.savemat(test_score_file, {'score_embed': score_embed.numpy(),
                                  'score': score.numpy(),
                                  'prob_list': test_prob_list,
                                  'prob_ids': test_prob_ids,
                                  'prob_cams': test_prob_cams},
                oned_as='column',
                do_compression=True)

    batch_size = 256
    num_probs = score.size(0)
    for i in range(0, num_probs, batch_size):
        test_score_file = osp.join(args.exp_dir, '%s_query_index_%d.mat' % (args.testset, i // batch_size))
        sio.savemat(test_score_file, {'prob_score': prob_score[i : i + batch_size],
                                    'index_in_gal': index_in_gal[i : i + batch_size]},
                    oned_as='column',
                    do_compression=True)

    test_time = time.time() - t0
    print("Total testing time: %.3f sec.\n" % test_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QAConv")
    # data
    parser.add_argument('--testset', type=str, default='market')
    parser.add_argument('--height', type=int, default=384,
                        help="input height, default: 384")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=resmap.names())
    parser.add_argument('--final_layer', type=str, default='layer3')
    parser.add_argument('--neck', type=int, default=512,
                        help="number of bottle neck channels, default: 512")
    parser.add_argument('--ibn', type=str, choices={'a', 'b', 'none'}, default='b', help="IBN type. Choose from 'a' or 'b'. Default: 'b'")
    parser.add_argument('--nhead', type=int, default=1,
                        help="the number of heads in the multiheadattention models (default=1)")
    parser.add_argument('--num_trans_layers', type=int, default=3,
                        help="the number of sub-encoder-layers in the encoder (default=3)")
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help="the dimension of the feedforward network model (default=2048)")
    parser.add_argument('--dropout', type=float, default=0., help="dropout, default: 0.")
    # test configs
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation only, default: False")
    parser.add_argument('-j', '--workers', type=int, default=8,
                        help="the number of workers for the dataloader, default: 8")
    parser.add_argument('--test_fea_batch', type=int, default=64,
                        help="Feature extraction batch size during testing. Default: 64."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--test_gal_batch', type=int, default=4,
                        help="QAConv gallery batch size during testing. Default: 4."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--test_prob_batch', type=int, default=4096,
                        help="QAConv probe batch size (as kernel) during testing. Default: 4096."
                             "Reduce this if you encounter a GPU memory overflow.")
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--exp-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'Exp'))

    main(parser.parse_args())
