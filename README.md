# MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://liuxuannan.github.io/MMFakeBench.github.io/)
[![arXiv](https://img.shields.io/badge/ArXiv-2403.01988-brightgreen)](https://arxiv.org/abs/2406.08772)


This is the official code repository of the MMFakeBench dataset. 

## Data repository

You should strictly follow the data usage guidelines by filling in [Data Protocol](https://docs.google.com/forms/d/e/1FAIpQLScKkQXn0uGN5Uu8oqFf4tU4NDU4scB-nMmTIPLoSEMxAeNwNA/viewform?edit_requested=true) and the download link will be sent to you once the form is accepted. 

### Annotations
Each iamge-text sample in the dataset is provided with annotations. For example, the annotation of a multimodal misinformation sample with mixed-source type may look like this in the MMFakeBench_val.json/MMFakeBench_test.json file:

```
{
    "image_path": "/fake/fever_AI_val_100/fever_dalle_val_1.jpg",
    "text": "Wi-Fi is a trademark of Microsoft.",
    "fake_cls": "textual_veracity_distortion",
    "gt_answers": ["Fake"]
    }
```

Where `image_path` is the relative path of the manipulated image, `text` is the manipulated text caption, `fake_cls` indicates the multiclass label of misinformation source, and `gt_answers` is the binary label of misinformation type.

## Data Structure
This dataset is structured around mixed-source multimodal misinformation detection.

```
.
├── MMFakeBench_val
│   ├── source
│   │   ├── MMFakeBench_val.json
│   ├── real
│   │   ├── bbc_val_50
│   │   │   ├── BBC_val_0000_002.jpg
│   │   │   ├── BBC_val_0067_219.jpg
│   │   │   └── ...
│   │   ├── guardian_val_50
│   │   │   ├── guardian_val_0800_000.jpg
│   │   │   ├── guardian_val_0809_691.jpg
│   │   │   └── ...
│   │   ├── usa_today_val_50
│   │   │   ├── usa_today_val_0002_156.jpg
│   │   │   ├── usa_today_val_0333_272.jpg
│   │   │   └── ...
│   │   ├── wash_val_50
│   │   │   ├── wash_val_0010_771.jpg
│   │   │   ├── wash_val_0057_599.jpg
│   │   │   └── ...
│   │   ├── fakeddit_val_50
│   │   │   ├── 1lcojw.jpg
│   │   │   ├── 1ncm16.jpg
│   │   │   └── ...
│   │   ├── coco_val_50
│   │   │   ├── coco_val_000000066771.jpg
│   │   │   ├── coco_val_000000078748.jpg
│   │   │   └── ...
│   ├── fake
# textual veracity distortion
│   │   ├── fever_AI_val_100
│   │   │   ├── fever_dalle_val_1.jpg
│   │   │   ├── fever_val_SD_1.jpg
│   │   │   ├── fever_val_AI_1.jpg
│   │   │   └── ...
│   │   ├── politicat_match_val_50
│   │   │   ├── politicat_match_val_0.jpg
│   │   │   ├── politicat_match_val_1.jpg
│   │   │   └── ...
│   │   ├── gossipcop_match_val_25
│   │   │   ├── gossipcop_match_val_0.jpg
│   │   │   ├── pgossipcop_match_val_1.jpg
│   │   │   └── ...
│   │   ├── gossipcop_midjourney_val_25
│   │   │   ├── gossipcop_val_1.jpg
│   │   │   ├── gossipcop_val_2.jpg
│   │   │   └── ...
│   │   ├── chatgpt_match_val_50
│   │   │   ├── chatgpt_match_val_0.jpg
│   │   │   ├── chatgpt_match_val_1.jpg
│   │   │   └── ...
│   │   ├── llm_gossip_md_generation_val_10
│   │   │   ├── llm_gossip_val_1.jpg
│   │   │   ├── llm_gossip_val_2.jpg
│   │   │   └── ...
│   │   ├── llm_science_md_generation_val_10
│   │   │   ├── llm_science_val_1.jpg
│   │   │   ├── llm_science_val_2.jpg
│   │   │   └── ...
│   │   ├── llm_rewrite_val_30
│   │   │   ├── chatgpt_rewrite_md_val_1.png
│   │   │   ├── chatgpt_rewrite_dalle_val_3.jpg
│   │   │   └── ...
# visual veracity distortion
│   │   ├── Fakeddit_photo_edit_val_50
│   │   │   ├── Fakeddit_photo_edit_val_0.jpeg
│   │   │   ├── Fakeddit_photo_edit_val_1.jpeg
│   │   │   └── ...
│   │   ├── antifact_image_generation_val_50
│   │   │   ├── coco_antifact_val_1.png
│   │   │   ├── coco_antifact_val_2.png
│   │   │   └── ...
# cross-modal consistency distortion
│   │   ├── Newsclipings_person_val_50
│   │   │   ├── Newsclipings_person_val_0.jpeg
│   │   │   ├── Newsclipings_person_val_1.jpeg
│   │   │   └── ...
│   │   ├── Newsclipings_scene_val_50
│   │   │   ├── Newsclipings_scene_val_0.jpeg
│   │   │   ├── Newsclipings_scene_val_1.jpeg
│   │   │   └── ...
│   │   ├── Newsclipings_semantic_val_50
│   │   │   ├── Newsclipings_semantic_val_0.jpeg
│   │   │   ├── Newsclipings_semantic_val_1.jpeg
│   │   │   └── ...
│   │   ├── DGM4_text_edit_senti_val_50
│   │   │   ├── DGM4_text_edit_senti_val_0.jpg
│   │   │   ├── DGM4_text_edit_senti_val_1.jpg
│   │   │   └── ...
│   │   ├── coco_text_edit_val_50
│   │   │   ├── coco_text_edit_val_2431_3_img_0.jpg
│   │   │   ├── coco_text_edit_val_8762_3_img_0.jpg
│   │   │   └── ...
│   │   ├── coco_text_edit_val_50
│   │   │   ├── coco_image_edit_val_1296_4_img_1.jpg
│   │   │   ├── coco_image_edit_val_5992_4_img_1.jpg
│   │   │   └── ...
```


## License
This dataset is under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.


# Citation:
If you found MMFakeBench useful in your research or applications, please kindly cite using the following BibTeX:
```
@article{liu2024mmfakebench,
  title={MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs},
  author={Liu, Xuannan and Li, Zekun and Li, Peipei and Xia, Shuhan and Cui, Xing and Huang, Linzhi and Huang, Huaibo and Deng, Weihong and He, Zhaofeng},
  journal={arXiv preprint arXiv:2406.08772},
  year={2024}
}
```