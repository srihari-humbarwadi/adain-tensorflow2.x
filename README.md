# AdaIN

## Getting started
 - Download the coco2017 and wikiart datasets
 - Generate the tfrecords for training and validation.
 ```
 python3 -m adain.dataset_utils.create_tfrecords --image_paths_pattern coco/train2017/* --prefix coco-train  --output_dir tfrecords
 python3 -m adain.dataset_utils.create_tfrecords --image_paths_pattern coco/val2017/* --prefix coco-val  --output_dir tfrecords
 python3 -m adain.dataset_utils.create_tfrecords --image_paths_pattern wikiart/train/* --prefix wikiart-train  --output_dir tfrecords
 ```
  - Start training with:
    - `python3 -m adain.main --config_path configs/coco-wikiart.json`
  - To export `saved_model`, use 
    - `python3 -m adain.export --config_path configs/coco-wikiart.json`

### Inference
```python
content_images = glob('assets/images/content/*')
style_images = glob('assets/images/style/*')

saved_model = tf.saved_model.load('export')
inference_fn = saved_model.signatures['serving_default']

content_image = read_image(content_images[3])
style_image = read_image(style_images[16])

alpha = tf.constant(1.0)
resize = tf.constant(True)

serving_input = {
    'style_images': style_image,
    'content_images': content_image,
    'alpha': alpha,
    'resize': resize
}

stylized_image = inference_fn(**serving_input)['stylized_images'][0]
result = prepare_visualization_image(
    content_image[0], 
    style_image[0], 
    stylized_image, figsize=(20, 5))

imshow(result, figsize=(20, 10))
```

### Results
![1.png](/assets/images/1.png)
![2.png](/assets/images/2.png)
![3.png](/assets/images/3.png)
![4.png](/assets/images/4.png)
![5.png](/assets/images/5.png)

##### Controlling Content-Style tradeoff by varying alpha
<table>
  <tr>
    <td valign="top"><img src="assets/gifs/chicago_asheville.gif"></td>
    <td valign="top"><img src="assets/gifs/lenna_en_campo_gris.gif"></td>
    <td valign="top"><img src="assets/gifs/avril_impronte_d_artista.gif"></td>
  </tr>
 </table>
</br>
</br>
</br>


### Training Logs
![tensorboard.png](/assets/images/tensorboard.png)

#### Tensorboard.Dev
[tensorboard.dev](https://tensorboard.dev/experiment/mK9V0nYtR06rGLu0Wvx7Tg/#scalars)


</br>
</br>
</br>

```

@article{DBLP:journals/corr/HuangB17,
  author    = {Xun Huang and
               Serge J. Belongie},
  title     = {Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization},
  journal   = {CoRR},
  volume    = {abs/1703.06868},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.06868},
  archivePrefix = {arXiv},
  eprint    = {1703.06868},
  timestamp = {Mon, 13 Aug 2018 16:46:12 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HuangB17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
