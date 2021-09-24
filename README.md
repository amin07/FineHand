# FineHand
Semi-automatic hand shape learning from videos in the context of sign language recognition <br>
Here is the diagram of the overall architecture, <br><br>
<img src="repo_imgs/arch.JPG" width="600" height="400" />
<br><br><br>
The project is divided into trainning two neural networks. One network is hand shape CNN responsible for learning hand shape patterns from images, the other one is a recurrent neural network (RNN) variation, which takes per frame hand shape representation and learns different signs.
## Paper/Cite
https://www.computer.org/csdl/proceedings-article/fg/2020/307900a397/1kecIh5NpVC
```
@INPROCEEDINGS {fineHandHosain,
author = {A. Hosain and P. Santhalingam and P. Pathak and H. Rangwala and J. KoÅ¡eckÃ¡},
booktitle = {2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020) (FG)},
title = {FineHand: Learning Hand Shapes for American Sign Language Recognition},
year = {2020},
pages = {397-404},
keywords = {deep learning;sign language;cnn;rnn},
doi = {10.1109/FG47880.2020.00062},
url = {https://doi.ieeecomputersociety.org/10.1109/FG47880.2020.00062},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {may}
}
```
## Data Download
* The running scripts below require two sets of data: annotated hand patch image for Network 1 and embedding from video hand patch for Network 2 <br>
* Two data direcotries are given as option ```-hdir``` and ```-dd``` 
* These two option refer to the directory ```data_iters``` and ```cropped_handpatches``` respectively
* The annotated hand patch download link: [annotated hand patch image download](https://drive.google.com/file/d/1BBwRGU8W17TK_eU_28y51c1Q-O7HqUmU/view?usp=sharing)
  *   contains three subdirectories: iter0, iter1, iter2
  *   each of the subdirectories contains 41 hand-shape class folders
  *   each of these folders contains the hand-patch sample images
* The cropped hand patch video link: [hand patch video download](https://drive.google.com/file/d/12mclaJTzQxkP7ZHfh9t7a3Btkf6YPAnf/view?usp=sharing)
  *   for each ASL video in GMU-ASL51, this folder contains two videos: one for left hand cropped patch and other for right hand
  *   an example hand video name is ```right_hand-wakeup_subject08_8_rgb```, meaning the sign video is from subject8 of sign class ```wakeup``` and for this video is for right hand

## Set up running environment
Run following commands after cloning the repo,
```
conda env create -f environments.yml
conda activate finehand_env
```

## Network 1 : Hand shape network
This is a image recognition type convolutional neural netowrk (CNN). An instance of ResNet50 was used here. Any other compatible CNN can be used. The goal of this CNN is to learn hand shape patterns as shown below. The per frame learned representation will be used later in the sign recognition phase.

Explanation of the options

* -hdir: directory of the hand shape obtained in each of the iterations (contains three iterations for our experiment)
* --test_subj: the subject for which we ignore all the training hand shape image because, this is the test subject
* --save_model: bool option, if specified, the script will save the trained cnn based on eval accuracy on test hand patches
* -ct: train type using both hands or single hand options are [both_hand, left_hand, right_hand]
* -lr: learning rate

### Training 
* ```python run_handshape_model.py -hdir <hand_shape_image_directory>  --test_subj <subject identifier> --save_model```
* **An example run:** ```python run_handshape_model.py -hdir data_iters/iter2/ -ts subject03```
### Evaluation
* ```python run_handshape_model.py -hdir <hand_shape_image_directory>  --test_subj subject03 -rm test -tm <full_location_to_trained_model>```
* **Example run:** ```python run_handshape_model.py -hdir  data_iters/iter2/ -ts subject03 -tm saves/handshape-model-subject03 -rm test```


## Network 2 : Recurrent Sign recognition network

In this training, the input hand-patch videos are used to extract hand features using the trained model in the previous step. The all hand feature embeddings are first save into temporary direcotry and the sign recognition models are trained on those embeddings. Finally, the temporary directory is being cleaned.

### Training and Evaluation
The lstm based sign recognition model will be trained and show maximum test accuracy on following commands,
* ```python run_lstm_sign_model.py -hcnn <saved_handshape_model_location> -dd <cropped_hand_video_direcotry> -bs <batch_size> -sr <sample_rate> -lr <learning_rate> -ts <test_subject> -ct <both hand vs single hand>```
* **An example run:** ```python run_lstm_sign_model.py -hcnn saves/handshape-model-subject03 -dd cropped_handpatches/ -bs 8 -sr 20 -lr 0.0001 -ts subject03 -ct both_hand```

## Contact
website : https://amin07.github.io/<br>
email : ahosain AT gmu DOT edu
