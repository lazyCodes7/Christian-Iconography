# Christian-Iconography
![Screenshot from 2022-01-08 18-26-30](https://user-images.githubusercontent.com/53506835/148644986-4e29f4df-fee0-47da-a289-b9ff3c589c41.png)

A computer vision pipeline to identify the "icons" in Christian paintings.

## A bit about iconography.
Iconography is related to identifying the subject itself in the image. So, for instance when I say Christian Iconography I would mean that I am trying to identify some objects like crucifix or mainly in this project the saints!

## Inspiration
I was looking for some interesting problem to solve and I came across RedHenLab's barnyard of projects and it had some really wonderful ideas there and this particular one intrigued me. On the site they didn't have much progress on it as the datasets were not developed on this subject but after surfing around I found something and just like that I got started!


## Dataset used.
The project uses the ArtDL dataset which contains 42,479 images of artworks portraying Christian saints, divided in 10 classes: Saint Dominic (iconclass 11HH(DOMINIC)), Saint Francis of Assisi (iconclass 11H(FRANCIS)), Saint Jerome (iconclass 11H(JEROME)), Saint John the Baptist (iconclass 11H(JOHN THE BAPTIST)), Saint Anthony of Padua (iconclass 11H(ANTONY OF PADUA), Saint Mary Magdalene (iconclass 11HH(MARY MAGDALENE)), Saint Paul (iconclass 11H(PAUL)), Saint Peter (iconclass 11H(PETER)), Saint Sebastian (iconclass 11H(SEBASTIAN)) and Virgin Mary (iconclass 11F). All images are associated with high-level annotations specifying which iconography classes appear in them (from a minimum of 1 class to a maximum of 7 classes).

## Sources
![Screenshot from 2022-01-08 18-08-56](https://user-images.githubusercontent.com/53506835/148644482-f3747702-2508-499c-b034-d97e790b9e52.png)

## Preprocessing steps.
All the images were first padded so that the resolution is sort of intact when the image is resized. A dash of normalization and some horizontal flips and the dataset is ready to be eaten/trained on by our model xD.

## Architecture used.
As mentioned the ArtDL dataset has around 43k images and hence training it completely wouldn't make sense. Hence a ResNet50 pretrained model was used.

But there is a twist.

Instead of just having the final classifying layer trained we only freeze the initial layer as it has gotten better at recognizing patterns from a lot of images it might have trained on. And then we fine-tune the deeper layers so that it learns the art after the initial abstraction. Another deviation is to replace the final linear layer by 1x1 conv layer to make the classification.

## Quantiative Results.

### Training
I trained the network for 10 epochs which took around 3 hours and used Stochastic Gradient Descent with LR=0.01 and momentum 0.9. The accuracy I got was 64% on the test set which can be further improved.

### Classification Report
![Screenshot from 2022-01-10 22-07-52](https://user-images.githubusercontent.com/53506835/148803160-ad659e9d-d48a-4fd4-8bf7-f943d000f3a7.png)

From the classification report it is clear that Saint MARY has the most number of samples in the training set and the precision for that is high. On the other hand other samples are low in number and hence their scores are low and hence we can't infer much except the fact that we need to oversample some of these classes so that we can gain more meaningful resuls w.r.t accuracy and of course these metrics as well

## Qualitative Results

We try an image of Saint Dominic and see what our classifier is really learning.

![Screenshot from 2022-01-10 22-10-37](https://user-images.githubusercontent.com/53506835/148803610-4675d71c-d4ef-4d75-a67c-5ed9ece2d270.png)

### Saliency Map
![Screenshot from 2022-01-10 22-12-31](https://user-images.githubusercontent.com/53506835/148803868-2af58632-9708-4595-bf5b-54b98cf543d6.png)

We can notice that regions around are more lighter than elsewhere which could mean that our classifier at least knows where to look :p

### Guided-Backpropagation
![Screenshot from 2022-01-10 22-14-26](https://user-images.githubusercontent.com/53506835/148804169-344c1c23-d0ae-4389-a88f-4b430ae43f4f.png)

So what really guided backprop does is that it points out the positve influences while classifiying an image. From this result we can see that it is really ignoring the padding applied and focussing more on the body and interesting enough the surroundings as well

### Grad-CAM!
![Screenshot from 2022-01-10 22-15-27](https://user-images.githubusercontent.com/53506835/148804850-5e358206-8a89-4229-aab8-9fd197b88562.png)

As expected the Grad-CAM when used shows the hot regions in our images and it is around the face and interesting enough the surrounding so maybe it could be that surroundings do have a role-play in type of saint?

## Possible improvements.
- Finding more datasets
- Or working on the architecture maybe?
- Using GANs to generate samples and make classifier stronger

## Citations
```
@misc{milani2020data,
title={A Data Set and a Convolutional Model for Iconography Classification in Paintings},
author={Federico Milani and Piero Fraternali},
eprint={2010.11697},
archivePrefix={arXiv},
primaryClass={cs.CV},
year={2020}
}
```
[RedhenLab's barnyard of projects](https://www.redhenlab.org/home/the-cognitive-core-research-topics-in-red-hen/the-barnyard/christian-iconography-the-emile-male-pipeline)
