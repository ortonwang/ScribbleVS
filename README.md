
### Introduction
This repository is for our paper: 'ScribbleVS: Scribble-Supervised Medical Image Segmentation via Regional Pseudo Labels Diffusion and Dynamic Selection'.

#Data:
You could refer https://github.com/HUANGLIZI/ScribbleVC or https://github.com/BWGZK/CycleMix for the preprocessed ACDC Dataset.

#Training:
put the dataset in the ACDC folder, 
--Data
----ACDC

```
python train.py  --gpu 0  --exp model1
python test_3D.py --exp model1 --gpu 0
```

### Acknowledgements:
Our code is origin from [ScribbleVC](https://github.com/HUANGLIZI/ScribbleVC) and [CycleMix](https://github.com/BWGZK/CycleMix).
Thanks for these authors for their valuable works.
