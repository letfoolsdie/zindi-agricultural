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

 ### A bit more details on models:
 [lvl_0_resnet50_augmented.ipynb](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/lvl_0_resnet50_augmented.ipynb) -- resnet50 with baseline spectrograms and added augmentations, training time ~1hour (all training time is measured on my machine with one 1080Ti); **Public leaderboard score 1.29**

 [lvl_0_densenet161_augmented.ipynb](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/lvl_0_densenet161_augmented.ipynb) -- densenet161 trained with about the same params as the resnet50 above, training time ~1hour. **Public leaderboard score 1.36**

 [lvl_0_SK_custom_specs.ipynb](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/lvl_0_SK_custom_specs.ipynb) -- a model architecture taken from [here](https://github.com/lRomul/argus-freesound/blob/master/src/models/simple_kaggle.py), dubbed simple_kaggle model. Uses different algorithm for generating spectrograms. Training time ~1.5hours. **Public leaderboard score 1.33**

 [lvl_0_resnet_melspec_mixup.ipynb](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/lvl_0_resnet_melspec_mixup.ipynb) -- use resnet34 to train on mel-spectrograms. Added mixup. Also started using torchaudio, so training time is reduced significantly. This model is trained on 3 folds with further test predictions averaging. Training time is ~10 minutes per fold, 30 minutes in total. **Public lb score 1.14**

[lvl_1_densenet161.ipynb](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/lvl_1_densenet161.ipynb) -- densenet161 trained on custom spectrograms with mixup. Train on 3 folds, average test predictions. Training time is ~2.15hours per fold, ~6.5hours in total. **Public lb score 0.99**

[lvl_1_dens161_aug_mixup.ipynb](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/lvl_1_dens161_aug_mixup.ipynb) -- about the same densenet as above but add augmentations. Total training time ~7.5hours, **public lb score 0.875**

[lvl_1_resneXt_aug_mixup.ipynb](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/lvl_1_resneXt_aug_mixup.ipynb) -- resnext50 trained with mixup and augmentations on 3 folds. Total training time ~6hours, **public lb score 0.90**

For the final submission all models' predictions generated in notebooks with names like `lvl_0_*` is first [averaged](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/1_average_submissions.ipynb) into one file (geometric mean). Then this file is [averaged](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/1_average_submissions.ipynb) with predictions from more heavy models (`lvl_1_*`). After that postprocessing is [applied](https://github.com/letfoolsdie/zindi-agricultural/blob/master/src/3_post_process_subm.ipynb)

### **Enviroment used:**
 - python 3.6.9
 - cuda 11.0
 - libraries from requirements.txt
 - trained on one 1080Ti



### Reproduce steps:

 - copy all audio files in `data/audio_files_full`;
 - run `0_preprocess_audios.ipynb` (converts all audios to wav with 22050 sample rate);
 - run notebooks with titles `lvl_0_*`, `lvl_1_*` in any order to train models and get predictions for test set;
 - (optional) run notebook `audioset_tagging_cnn/2_get_speech_prediction.ipynb` to use pretrained PANN to predict probability of speech for each audio frame by frame (need to download pretrained PANN weights before). These predictions will then be used in postprocessing of the final submission. It's optional since I've included the result file in this repo (probs_dict.joblib)
 - Run `1_average_submissions.ipynb` to compute geometric mean of generated submissions;
 - Run `3_post_process_subm.ipynb` for postprocessing of final submission. This notebook filters out probably invalid test files (the ones with speech probability below threshold) and replaces predictions for them by a constant based on each class frequency in train data


I also wrote a kind of lab journal during the competition, making notes of ideas I have and logging almost all submissions I've made. It proved to be quite usefull. I'll leave a link [here](https://docs.google.com/spreadsheets/d/1XHxWI-XcxGEwyPQSjWkYzG1dzIxCQcUuPGTjhenrDDU/edit?usp=sharing) for those interested, but I should warn you, it's written in a wild mix of russian and english :)