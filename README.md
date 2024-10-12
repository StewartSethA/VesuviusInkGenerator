<h1>Vesuvius Ink Generator</h1>

The Vesuvius Ink Generator is an attempt to use the [Activation Maximization technique](https://towardsdatascience.com/reveling-what-neural-networks-see-and-learn-pytorchrevelio-a218ef5fc61f) and other deep learning methods to **generate artificial ink images** starting from random noise.
The idea is simply to start with a random cube of volumetric noise data and backpropagate the signal from a pretrained ink model to create a new ink image.

In this way, we can visualize **what the model has been trained to look for**, and directly visualize signals that look maximally (or minimally) "inky".

So far there are four primary modes of use:
1. Maximize ink signal starting from a random noise volume
2. Maximize ink signal starting from an existing scroll volume,
3. Minimize ink signal starting from a random noise image,
4. Minimize ink signal starting from an existing scroll volume

We can also do diffs to isolate the changes made to a random block or scroll segment that would maximize or minimize the ink signal.

<h2>How to Generate Ink Images</h2>
To generate random ink images, run

```python
python generaterandomink.py
```
The dataloader expects there to be an ```eval_scrolls``` directory and a ```train_scrolls``` directory containing the segment(s) to be used.

TODO: Add random-only mode that requires no data to be downloaded

<h2>Generated sample images</h2>
Visual for slices of a 3D volume, optimized starting from random noise to maximize the appearance of ink:

![Slice of a randomly generated ink volume](https://github.com/StewartSethA/VesuviusInkGenerator/blob/master/samples/randinkblock.png)

<img src="https://github.com/StewartSethA/VesuviusInkGenerator/blob/master/samples/[2720, 128, 2784, 192]_25z_99.ink.png">

Link to download generated ink volumes (as stacks of .png files):
[Download .zip of generated sample volumes](https://drive.google.com/file/d/173qsupr1McDwvVuHe2lBKoN8WfAQLDaJ/view?usp=sharing)

This repo is based off of inference code from [Bodillium's fork](https://github.com/Bodillium/Vesuvius-Grandprize-Winner) of the [2023 GP winner by younader](https://github.com/younader/Vesuvius-Grandprize-Winner/forks).
