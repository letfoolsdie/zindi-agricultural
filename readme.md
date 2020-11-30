This repo contains solution for 6th place of GIZ NLP Agricultural Keyword Spotter competiton
(https://zindi.africa/competitions/giz-nlp-agricultural-keyword-spotter/leaderboard)


### **Final solution overview:**
Geometric mean of several models trained on 3-fold cross-validation.
Final submission includes:
 - resnet50 trained with augmentations (gaussian noise, pitch shift, time stretch) and mixup;
 - densenet161 trained with mixup;
 - densenet161 trained with mixup and augmentations (gaussian noise and pitch shift)
 - resneXt50 also trained with mixup and noise/pitch shift augmentations
 - geometric mean of several simpler/worse-performing models (models from notebooks starting with `lvl_0_*`)

I've also used post-processing based on pretrained PANNs to replace predictions for junk test audios with simple constant based on class frequency.

### **Enviroment used:**
 - python 3.6.9
 - cuda 11.0
 - libraries from requirements.txt
 - trained on single 1080Ti



### Reproduce steps:

 - copy all audio files in `data/audio_files_full`;
 - run `0_preprocess_audios.ipynb` (converts all audios to wav with 22050 sample rate);
 - run notebooks with titles `lvl_0_*`, `lvl_1_*` in any order to train models and get predictions for test set;
 - (optional) run notebook `audioset_tagging_cnn/2_get_speech_prediction.ipynb` to use pretrained PANN to predict probability of speech for each audio frame by frame (need to download pretrained PANN weights before). These predictions will then be used in postprocessing of the final submission. It's optional since I've included the result file in this repo (probs_dict.joblib)
 - Run `1_average_submissions.ipynb` to compute geometric mean of generated submissions;
 - Run `3_post_process_subm.ipynb` for postprocessing of final submission. This notebook filters out probably invalid test files (the ones with speech probability below threshold) and replaces predictions for them by a constant based on each class frequency in train data