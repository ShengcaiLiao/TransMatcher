% Demo of the QAConv matching.
% For the score matrix, probe is in rows, and gallery is in columns.
% Tested on Octave 4.4.1.

close all; clear; clc;

% modify the followings according to the test_matching.py
dataset = 'market';
% prob_range = '0-1000';
model_dir = './';
% score_file =[model_dir, '/', dataset, '_query_score_', prob_range, '.mat'];
score_file =[model_dir, '/', dataset, '_query_score.mat'];
img_dir = 'Market-1501-v15.09.15\\query/';

threshold = 0.5;
num_match_threshold = 5;
height = 384;
width = 128;

% out_dir = sprintf('%s%s%s_thr=%g/', model_dir, dataset, prob_range, threshold);
out_dir = sprintf('%s%s_thr=%g/', model_dir, dataset, threshold);
pos_out_dir = [out_dir, 'positives/'];
neg_out_dir = [out_dir, 'negatives/'];
mkdir(out_dir);
mkdir(pos_out_dir);
mkdir(neg_out_dir);

load(score_file, 'score_embed', 'score', 'match_index', 'prob_ids', 'prob_cams', 'prob_list');

[num_probs, ~, num_layers, ~, hei, wid] = size(match_index);
prob_score = reshape(match_index(:, :, :, 1, :, :), [num_probs, num_probs, num_layers, hei, wid]);
index_in_gal = reshape(match_index(:, :, :, 2, :, :), [num_probs, num_probs, num_layers, hei, wid]);
gal_score = reshape(match_index(:, :, :, 3, :, :), [num_probs, num_probs, num_layers, hei, wid]);
index_in_prob = reshape(match_index(:, :, :, 4, :, :), [num_probs, num_probs, num_layers, hei, wid]);

% scale matching scores to make them visually more recognizable
prob_score = prob_score * 200;

% num_probs = size(prob_score)(1);
prob_ids = prob_ids(1:num_probs);
prob_cams = prob_cams(1:num_probs);
prob_list = prob_list(1:num_probs);

for i = 1 : num_probs
  score(i,i) = 0;
end

images = cell(num_probs, 1);

for i = 1 : num_probs
  filename = prob_list{i};
  images{i} = imread([img_dir, filename]);
end

for i = 1 : num_probs
  sam_index = find(prob_ids == prob_ids(i) & prob_cams ~= prob_cams(i));
  num_sam = length(sam_index);
  
  for j = 1 : num_sam
    if j == i
      continue;
    end

    index_j = sam_index(j);
    file_i = prob_list{i};
    file_j = prob_list{index_j};

    for k = 1 : num_layers
      [num_matches(k), img(k)] = draw_lines(images, height, width, prob_score, index_in_gal, i, index_j, k, threshold);
    end
    
    fprintf('Probe %d: positive, score=%g, #matches=(%d, %d, %d).\n', i, score(index_j, i), num_matches(1), num_matches(2), num_matches(3));
      
    if min(num_matches) >= num_match_threshold
      filename = sprintf('%s/%d_%.4f_%s-%s', pos_out_dir, sum(num_matches), score(index_j, i), file_i(1:end-4), file_j);
      imwrite([img(1), img(2), img(3)], filename);
    end
  end
          
  sam_index = find(prob_ids ~= prob_ids(i) & prob_cams ~= prob_cams(i));
  num_sam = length(sam_index);
  
  sam_score = score(sam_index, i);
  [max_score, max_index] = max(sam_score);
  
  index_j = sam_index(max_index);
  file_j = prob_list{index_j};
  
  for k = 1 : num_layers
    [num_matches(k), img(k)] = draw_lines(images, height, width, prob_score, index_in_gal, i, index_j, k, threshold);
  end

  fprintf('\t negative, max score=%g, #matches=(%d, %d, %d).\n', max_score, num_matches(1), num_matches(2), num_matches(3));  
  
  if min(num_matches) >= num_match_threshold
    filename = sprintf('%s%d_%.4f_%s-%s', neg_out_dir, sum(num_matches), max_score, file_i(1:end-4), file_j);
    imwrite([img(1), img(2), img(3)], filename);
  end
end
