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




