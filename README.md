# AdaIN

## Getting started
 - Download the coco2017 and wikiart datasets
 - Generate the tfrecords for training and validation.
 ```
 python -m adain.dataset_utils.create_tfrecords --image_paths_pattern coco/train2017/* --prefix coco-train  --output_dir tfrecords
 python -m adain.dataset_utils.create_tfrecords --image_paths_pattern coco/val2017/* --prefix coco-val  --output_dir tfrecords
 python -m adain.dataset_utils.create_tfrecords --image_paths_pattern wikiart/train/* --prefix wikiart-train  --output_dir tfrecords
 ```
  - Start training with `python -m adain.main --config_path configs/coco-wikiart.json`
  - To export `saved_model`, use `python -m adain.export --config_path configs/coco-wikiart.json`. The `saved_model` artifacts will be avaiable at `./export`
  
### Results
![1.png](/assets/images/1.png)
![2.png](/assets/images/2.png)

##### Controlling Content-Style tradeoff by varying alpha

![1_interpolation.png](/assets/images/1_interpolation.png)
![2_interpolation.png](/assets/images/2_interpolation.png)

</br>
</br>
</br>


### TensorBoard
![tensorboard.png](/assets/images/tensorboard.png)



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
