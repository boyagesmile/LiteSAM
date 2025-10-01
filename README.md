# LiteSAM

This project is for [LiteSAM: Lightweight and Robust Feature Matching for Satellite and Aerial Imagery](https://www.mdpi.com/2072-4292/17/19/3349)

---

## 📥 Model Download
Pretrained model weights can be downloaded from:  
[Google Drive Link](https://drive.google.com/file/d/1fheBUqQWi5f55xNDchumAx2-SmGdT-mX/view?usp=drive_link)

---

## ⚙️ Environment Setup

You can set up the environment in two different ways:

### Option 1 – Using the Conda YAML file
If you use **conda**, create the environment directly from the YAML file:

```bash
conda env create -f environment.yml
conda activate litesam
```
### Option 2 – Using requirements.txt

If you already have a Python environment prepared:
```bash
pip install -r requirements.txt
```

## 🚀 Testing / Running the Code

After installing the environment and downloading the pretrained model weights, you can run the evaluation:
```bash
python test_mega.py
```
This will start evaluation on the MegaDepth dataset.  
You can also modify the script to test on your own dataset if needed.

---

## 🙏 Acknowledgement
This project builds upon the excellent work of  
[EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR).
