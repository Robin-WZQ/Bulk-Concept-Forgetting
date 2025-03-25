# Bulk Concepts Forgetting for Text-to-Image Diffusion Models

<img style="witdh: 100%;" src="images/first_pic.png">

This is a project aiming to erase 10^3 concepts on T2I diffusion models while remaining its orginal performace, i.e., fidelity.

This project is not for scientific research purposes. JUST FOR FUN!!!

## Installation
```
conda create -n bcf python=3.10
conda activate bcf
cd bcf
pip install -r requirement.txt
```

## Run BCF

- Forgetting Artistic Styles

```
python clip_edit.py --file 'data/art_forget/artist_list_forget_10.csv' --algorithm 'bcf' --concept_type 'artist_forget' --seed 42 --v_similarity_metric "cosine"
```

- Forgetting Objects

```
python clip_edit.py --file 'data/object_forget/object_erase_10.csv' --algorithm 'bcf' --concept_type 'object_forget' --seed 42 --v_similarity_metric "cosine"
```

- Forgetting NSFW

```
python clip_edit.py --file 'data/nsfw_forget/nsfw_list_forget_all.csv' --algorithm 'bcf' --concept_type 'NSFW' --seed 42 --v_similarity_metric "cosine"
```

Forgetting results are stored at `images/results/bcf/<concept_type>/` .

Edited model are stored at `model/<concept_type>/` .



