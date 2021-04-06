# DG-Font: Deformable Generative Networks for Unsupervised Font Generation
The source code for 'DG-Font: Deformable Generative Networks for Unsupervised Font Generation', by Yangchen Xie, Xinyuan Chen, Li sun and Yue lu. The paper was accepted by CVPR2021.

Arvix version

# Dependencies

Libarary
-------------

    pytorch==1.1.0 or 1.2.0  
    tqdm  
    numpy
    opencv-python  
    scipy  
    sklearn
    matplotlib  
    pillow  
    tensorboardX 

DCN
--------------

please refer to https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0 to install the dependencies of deformable convolution.

Dataset
--------------
[方正字库](https://www.foundertype.com/index.php/FindFont/index) provides free font download for non-commercial users.

Example directory hierarchy

    Project
    |--- DG-Font
    |          |--- font2img.py    
    |          |--- main.py
    |          |--- train
    |                 |--- train.py
    |
    |--- data
           |--- font1
           |--- font2
                 |--- 0000.png
                 |--- 0001.png
                 |--- ...
           |--- ...



# How to run

prepare dataset

    python font2img.py --ttf_path ttf_folder --chara character.txt --save_path save_folder --img_size IMAGESIZE --chara_size CHARACTERSIZE

train

    python main.py --gpu GPU_ID --img_size IMAGESIZE --data_path /path/to --output_k CLASS_NUM --batch_size BATCHSIZE --val_num TEST_IMGS_NUM_FOR_EACH_CLASS

test

    python main.py --gpu GPU_ID --img_size IMAGESIZE --data_path /path/to --output_k CLASS_NUM --batch_size BATCHSIZE --validation --load_model $DIR_TO_LOAD
    
# Acknowledgements
We would like to thank @Johnson yue and 上海驿创信息技术有限公司 for their advices in code. Our code  is based on (TUNIT)(https://github.com/clovaai/tunit).


# Bibtex
    @inproceedings{DG-Font,
        title={DG-Font: Deformable Generative Networks for Unsupervised Font Generation},
        author={Yangchen Xie, Xinyuan Chen, Li sun, Yue lu},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2021}
    }
