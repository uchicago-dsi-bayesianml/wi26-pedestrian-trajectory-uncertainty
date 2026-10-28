# Data folder

The ETH/UCY trajectory files are not committed. The scripts and the safety notebook read them from `data/raw/raw/{train,val,test}/`, which is the team's consolidation of the leave-one-scene-out folders in the [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) repository (`experiments/pedestrians/raw/`, MIT license). Each file is tab-separated with four columns: frame, pedestrian id, x and y in metres, one row per annotated frame at 2.5 Hz.

To rebuild the folder:

1. Clone Trajectron++ anywhere, or download just `experiments/pedestrians/raw/`.
2. Copy every `*_train.txt` from any one of its five scene folders (`eth`, `hotel`, `univ`, `zara1`, `zara2` under `raw/`) into `data/raw/raw/train/`, and every `*_val.txt` into `data/raw/raw/val/`. The training and validation files for a scene are identical across the folders that contain them.
3. For the test folder, take each scene folder's single test file and rename it with a `_test` suffix: `eth/test/biwi_eth.txt` becomes `biwi_eth_test.txt`, and likewise `biwi_hotel`, `crowds_zara01`, `crowds_zara02`, `students001` and `students003` (the last two are both in `univ/test/`). `uni_examples` and `crowds_zara03` have no test file.

Note that a scene's test file is the whole scene: for ETH, for example, it is exactly the training file plus the validation file, row for row. The staff README explains what that means for the project's evaluation.

The original recordings are the BIWI Walking Pedestrians dataset (Pellegrini et al., ICCV 2009, ETH Zurich) and the Crowds-by-Example dataset (Lerner et al., Computer Graphics Forum 2007, University of Cyprus), published for research use and redistributed by Trajectron++ and Social GAN in this text format. Cite the original papers if you use them.
