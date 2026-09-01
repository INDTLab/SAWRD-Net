<p align="center">
  <img src="https://raw.githubusercontent.com/INDTLab/SAWRD-Net/main/assets/title.png" alt="image" width="70%">
</p>

<p align="center"> 
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320326017851" ><img src="https://img.shields.io/badge/HOME-PR-blue.svg"></a>
<a href="" ><img src="https://img.shields.io/badge/HOME-Paper-important.svg"></a>
<a href="https://www.researchgate.net/publication/413548602_Water_Reflection_Detection_Using_Symmetric_Attention/link/6a8beb2b3a7ce868e72f06f5/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoicHVibGljYXRpb24iLCJwcmV2aW91c1BhZ2UiOiJfZGlyZWN0In19" ><img src="https://img.shields.io/badge/PDF-Paper-blueviolet.svg"></a>

</p>

# Architecture
<p align="center">
  <img src="https://raw.githubusercontent.com/INDTLab/SAWRD-Net/main/assets/architecture.png" alt="image" width="70%">
</p>

# Usage
### Installation
1. Create the environment from the `environment.yaml` file:   
   ```copy
   conda env create -f environment.yaml
   ```     
2. Activate the new environment:  
   ```copy
   conda activate sawrdnet
   ```    
3. Verify that the new environment was installed correctly:  
    ```copy
   conda env list
    ```    

### Configuration

### Data Sets
Download Water Reflection Scene Data Set(WRSD) : <a href="https://drive.google.com/file/d/1D00quOYefmW_VoBnJVNjkezOa2w-aUWl/view?usp=drive_link">GoogleLink</a> or <a href="https://pan.baidu.com/s/1G8E_m03HXL2M6IszvoDLUw?pwd=6bmg ">BaiduLink</a> with code `6bmg`      

### Train and Test
You can use command like this:  
```copy
python SAWRD-Net.py
```
You can also download th pre-trained models on the WRSD: <a href="https://drive.google.com/file/d/1FCx4BShJsBLxDu9w4kwTBjKjdXWMFgNb/view?usp=drive_link">GoogleLink</a> or <a href="https://pan.baidu.com/s/1ZOg3o7DXqRnzMVOf6WQuEQ?pwd=adzl ">BaiduLink</a>  with code adzl, and then use command like this:  
```copy
python SAWRD-Net_test.py
```

### Calculate TP Rate
You can run evaluate.ipynb in Jupyter Notebook to calculate the TP Rate.


# Results

<div align=center><img src="https://raw.githubusercontent.com/INDTLab/SAWRD-Net/main/assets/results.png" width=60%></div>  
<div align=center><img src="https://raw.githubusercontent.com/INDTLab/SAWRD-Net/main/assets/results2.png" width=60%></div>  

# Citation
```
@article{article,
author = {Yao, Shuxuan and Wang, Chengjia and Sun, Jianyuan and Dong, Junyu and Dong, Xinghui},
year = {2026},
month = {08},
pages = {},
title = {Water Reflection Detection Using Symmetric Attention},
journal = {Pattern Recognition},
doi = {10.1016/j.patcog.2026.114821}
}
```
