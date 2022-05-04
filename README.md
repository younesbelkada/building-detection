# building-detection

Building segmentation tool using a UNet.
The code is inspired from [this](https://www.kaggle.com/code/yangpeng1995/building-extraction-in-dl-training-and-inference/notebook?fbclid=IwAR3kqbsSTbAa6t5Aldg2m6s_7IeutO3SJBD1U9zeE7SegjpJ6jlBpc6sJQI) kaggle challenge.

## Setup

First install the dependencies by doing:
```
pip install -r requirements.txt
```
Get the pretrained weights from [here](https://www.kaggleusercontent.com/kf/42175176/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..PV-_fSsZFHlETYIxgIZ2AA.1WCkwR_dNUT34-pI-MGC1UOlLjBvR6Fb-ozsb4JhUL9GYg0OkrT000CDhyR-5L1nvqcPh34Tyw3uMs2b9ArWEqK2p__oa5vklVBsYoc7pBmwZFAon-l2hWpJ9PD4-iai0statB7cPQp23YzVwqKreTL0jOyC_D6MWVJZaGsZoI_y46nTgKFtH4sQG3FniH10bz6TTZhy4vRmNZshZufoKdfRpOWC9sfv-hm9ke-jzHym8tOvllUlBfFH3zDrxO2uKkdFJiiY7Y4agbxHKb3SeKFDdXqBB3cnoTWrDp8C9Ltt0p22KJekrrEfHjz45nYd_pusu7KkSL8vVbbSbBNmI4Nt2AXOTHSFEmJ4vI0RPGYJU_amelzALqPTUtZxnnUY6nEm5liLQhctXdeYuSvd6IW11-ISz6VQC3473nZxBjHTIsTZ6hxnwLSEywmfhOop6Ng56ocw51Sk6bUUAe5iibMmEGkBR3WlaPbp4VOkCwMv92I1IGU-xP0oAPGJuohgRBlGRZ7cILgQ6Tbd3Ajld0oBFEryt5ctqwhsxljQ5qYib8Up1t2FCDZbVnVR14GQYuZc2ZkmZwNTQZDfkr1SeMYQxf1pdPvFIOFS9T6wAZNqvBFJXhatCYby9oJ3GjZ9onDf6iq9sWooEbQ2371aFso_xXp8ezKpar9jaw0FRnpyjqZ9kIErsd0y-vQJY5FRToCkKhk1hrgBwqP9sHiilg._23rmZb1EHArAsOLAd1O4A/model.pth) and put the file ```model.pth``` into a directory named ```weights/```.

## How to run some predictions?

Put your input images into ```images``` folder and simply run:
```
python run.py
```