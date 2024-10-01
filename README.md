Vesuvius Ink Generator

The Vesuvius Ink Generator is an attempt to use gradient ascent to generate artificial ink images starting from random noise.
The idea is simply to start with a random cube of volumetric noise data and backpropagate the signal from a pretrained ink model to create a new ink image.

![Slice of a randomly generated ink volume](https://github.com/StewartSethA/VesuviusInkGenerator/blob/master/randinkblock.png)

Link to download generated ink volumes (as stacks of .png files):
https://drive.google.com/file/d/173qsupr1McDwvVuHe2lBKoN8WfAQLDaJ/view?usp=sharing

This repo is based off of inference code from Bodillium's fork of the 2023 GP winner by younader.
