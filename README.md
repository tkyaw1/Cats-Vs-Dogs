# Cats-Vs-Dogs
Given a photo, our algorithm determines whether it's a photo of a cat or a dog. 

I worked on this project with Leah Brumgard for our final project in Artificial Intelligence.

HOW WE PREPROCESSED THE DATA:
We downloaded the data from the Cats Vs. Dogs competition on Kaggle (https://www.kaggle.com/c/dogs-vs-cats). We used 20,024
images from the train folder in the data set. We preprocessed the images by padding
the smaller images with zeros to create a black border so that all the images had the
same size. The final size of the images were 768 x 1050 pixels, each with RGB values.
We then normalized the RGB colors (0 as the min and 255 as the max).
After preprocessing all of the images, we wrote them into
40 separate files containing shuffled photos, along with one file
containing the corresponding labels for each photo.

For more information, please refer to the final write up pdf.

