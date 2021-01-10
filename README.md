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
## Network 1 : Hand shape network
This is a image recognition type convolutional neural netowrk (CNN). An instance of ResNet50 was used here. Any other compatible CNN can be used. The goal of this CNN is to learn hand shape patterns as shown below. The per frame learned representation will be used later in the sign recognition phase.

### Training 
```python run_handshape_model.py -hdir <hand_shape_image_directory>  --test_subj subject03 --save_model```
### Evaluation
```python run_handshape_model.py -hdir <hand_shape_image_directory>  --test_subj subject03 -rm test -tm <full_location_to_trained_model>```

## Network 2 : Recurrent Sign recognition network


### Training 

### Evaluation

