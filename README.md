# DG_Maxwell
Solution of Wave equation using Discontinuous Galerkin method.

## Getting Started

Clone the repository
```
git clone git@github.com:QuazarTech/DG_Maxwell.git
cd DG_Maxwell
```

## Dependencies:
* Arrayfire
* Numpy
* Matplotlib
* tqdm
* pytest

## Usage:
```
cd DG_Maxwell/code
python3 main.py
```
* The parameters of the simulation are stored in global_variables.py in
  the app folder, These can be changed accordingly.
  
* The images of the wave are stored in the folder 1D_wave_images folder.

* To stitch the images in the folder and obtain a video of the simulation,
  use the command in the terminal -
  `ffmpeg -f image2 -i %04d.png -vcodec mpeg4 -mbd rd -trellis 2 -cmp 2 -g 300 -pass 1 -r 25 -b 18000000 movie.mp4`
  
## Authors

* **Balavarun P** - [GitHub Profile](https://github.com/Balavarun5)
* **Aman Abhishek Tiwari** - [GitHub Profile](https://github.com/amanabt)
* **Mani Chandra** - [GitHub Profile](https://github.com/mchandra)

## Note for developers:
* Use tab spaces for indentation.
