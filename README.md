<div align="center">
<img alt="LOGO" src="https://avatars.githubusercontent.com/u/127122328?s=400&u=5395a98a4f945a3a50cb0cc96c2747505d190dbc&v=4" width="300" height="300" />
  
# SoftVC VITS Singing Voice Conversion

[**English**](./README.md) | [**中文简体**](./README_zh_CN.md)

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/svc-develop-team/so-vits-svc/blob/4.1-Stable/sovits4_for_colab.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-AGPL3.0-green.svg?style=for-the-badge)](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/LICENSE)

This round of limited time update is coming to an end, the warehouse will enter the Archieve state, please know

</div>

> ✨ A studio that contains visible f0 editor, speaker mix timeline editor and other features (Where the Onnx models are used) : [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio)

> ✨ A fork with a greatly improved user interface: [34j/so-vits-svc-fork](https://github.com/34j/so-vits-svc-fork)

> ✨ A client supports real-time conversion: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

**This project differs fundamentally from VITS, as it focuses on Singing Voice Conversion (SVC) rather than Text-to-Speech (TTS). In this project, TTS functionality is not supported, and VITS is incapable of performing SVC tasks. It's important to note that the models used in these two projects are not interchangeable or universally applicable.**

## Announcement

The purpose of this project was to enable developers to have their beloved anime characters perform singing tasks. The developers' intention was to focus solely on fictional characters and avoid any involvement of real individuals, anything related to real individuals deviates from the developer's original intention.

## Disclaimer

This project is an open-source, offline endeavor, and all members of SvcDevelopTeam, as well as other developers and maintainers involved (hereinafter referred to as contributors), have no control over the project. The contributors have never provided any form of assistance to any organization or individual, including but not limited to dataset extraction, dataset processing, computing support, training support, inference, and so on. The contributors do not and cannot be aware of the purposes for which users utilize the project. Therefore, any AI models and synthesized audio produced through the training of this project are unrelated to the contributors. Any issues or consequences arising from their use are the sole responsibility of the user.

This project is run completely offline and does not collect any user information or gather user input data. Therefore, contributors to this project are not aware of all user input and models and therefore are not responsible for any user input.

This project serves as a framework only and does not possess speech synthesis functionality by itself. All functionalities require users to train the models independently. Furthermore, this project does not come bundled with any models, and any secondary distributed projects are independent of the contributors of this project.

## 📏 Terms of Use

# Warning: Please ensure that you address any authorization issues related to the dataset on your own. You bear full responsibility for any problems arising from the usage of non-authorized datasets for training, as well as any resulting consequences. The repository and its maintainer, svc develop team, disclaim any association with or liability for the consequences. 

1. This project is exclusively established for academic purposes, aiming to facilitate communication and learning. It is not intended for deployment in production environments.
2. Any sovits-based video posted to a video platform must clearly specify in the introduction the input source vocals and audio used for the voice changer conversion, e.g., if you use someone else's video/audio and convert it by separating the vocals as the input source, you must give a clear link to the original video or music; if you use your own vocals or a voice synthesized by another voice synthesis engine as the input source, you must also state this in your introduction.
3. You are solely responsible for any infringement issues caused by the input source and all consequences. When using other commercial vocal synthesis software as an input source, please ensure that you comply with the regulations of that software, noting that the regulations of many vocal synthesis engines explicitly state that they cannot be used to convert input sources!
4. Engaging in illegal activities, as well as religious and political activities, is strictly prohibited when using this project. The project developers vehemently oppose the aforementioned activities. If you disagree with this provision, the usage of the project is prohibited.
5. If you continue to use the program, you will be deemed to have agreed to the terms and conditions set forth in README and README has discouraged you and is not responsible for any subsequent problems.
6. If you intend to employ this project for any other purposes, kindly contact and inform the maintainers of this repository in advance.

## 📝 Model Introduction

The singing voice conversion model uses SoftVC content encoder to extract speech features from the source audio. These feature vectors are directly fed into VITS without the need for conversion to a text-based intermediate representation. As a result, the pitch and intonations of the original audio are preserved. Meanwhile, the vocoder was replaced with [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) to solve the problem of sound interruption.

### 🆕 4.1-Stable Version Update Content

- Feature input is changed to the 12th Layer of [Content Vec](https://github.com/auspicious3000/contentvec) Transformer output, And compatible with 4.0 branches.
- Update the shallow diffusion, you can use the shallow diffusion model to improve the sound quality.
- Added Whisper-PPG encoder support
- Added static/dynamic sound fusion
- Added loudness embedding
- Added Functionality of feature retrieval from [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  
### 🆕 Questions about compatibility with the 4.0 model

- To support the 4.0 model and incorporate the speech encoder, you can make modifications to the `config.json` file. Add the `speech_encoder` field to the "model" section as shown below:

```
  "model": {
    .........
    "ssl_dim": 256,
    "n_speakers": 200,
    "speech_encoder":"vec256l9"
  }
```

### 🆕 Shallow diffusion
![Diagram](shadowdiffusion.png)

## 💬 Python Version

Based on our testing, we have determined that the project runs stable on `Python 3.8.9`.

## 📥 Pre-trained Model Files

#### **Required**

**You need to select one encoder from the list below**

##### **1. If using contentvec as speech encoder(recommended)**

`vec768l12` and `vec256l9` require the encoder

- ContentVec: [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  - Place it under the `pretrain` directory

Or download the following ContentVec, which is only 199MB in size but has the same effect:
- ContentVec: [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)
  - Change the file name to `checkpoint_best_legacy_500.pt` and place it in the `pretrain` directory

```shell
# contentvec
wget -P pretrain/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O checkpoint_best_legacy_500.pt
# Alternatively, you can manually download and place it in the hubert directory
```

##### **2. If hubertsoft is used as the speech encoder**
- soft vc hubert: [hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  - Place it under the `pretrain` directory

##### **3. If whisper-ppg as the encoder**
- download model at [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), the model fits `whisper-ppg`
- or download model at [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), the model fits `whisper-ppg-large`
  - Place it under the `pretrain` directory
  
##### **4. If cnhubertlarge as the encoder**
- download model at [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)
  - Place it under the `pretrain` directory

##### **5. If dphubert as the encoder**
- download model at [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)
  - Place it under the `pretrain` directory

##### **6. If WavLM is used as the encoder**
- download model at  [WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), the model fits `wavlmbase+`
  - Place it under the `pretrain` directory

##### **7. If OnnxHubert/ContentVec as the encoder**
- download model at [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
  - Place it under the `pretrain` directory

#### **List of Encoders**
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### **Optional(Strongly recommend)**

- Pre-trained model files: `G_0.pth` `D_0.pth`
  - Place them under the `logs/44k` directory

- Diffusion model pretraining base model file: `model_0.pt`
  - Put it in the `logs/44k/diffusion` directory

Get Sovits Pre-trained model from svc-develop-team(TBD) or anywhere else.

Diffusion model references [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) diffusion model. The pre-trained diffusion model is universal with the DDSP-SVC's. You can go to [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)'s repo to get the pre-trained diffusion model.

While the pretrained model typically does not pose copyright concerns, it is essential to remain vigilant. It is advisable to consult with the author beforehand or carefully review the description to ascertain the permissible usage of the model. This helps ensure compliance with any specified guidelines or restrictions regarding its utilization.

#### **Optional(Select as Required)**

##### NSF-HIFIGAN

If you are using the `NSF-HIFIGAN enhancer` or `shallow diffusion`, you will need to download the pre-trained NSF-HIFIGAN model.

- Pre-trained NSF-HIFIGAN Vocoder: [nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
  - Unzip and place the four files under the `pretrain/nsf_hifigan` directory

```shell
# nsf_hifigan
wget -P pretrain/ https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip -od pretrain/nsf_hifigan pretrain/nsf_hifigan_20221211.zip
# Alternatively, you can manually download and place it in the pretrain/nsf_hifigan directory
# URL: https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1
```

##### RMVPE

If you are using the `rmvpe` F0 Predictor, you will need to download the pre-trained RMVPE model.

+ download model at [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip), this weight is recommended.
  + unzip `rmvpe.zip`，and rename the `model.pt` file to `rmvpe.pt` and place it under the `pretrain` directory

- ~~download model at [rmvpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt)~~
  - ~~Place it under the `pretrain` directory~~

##### FCPE(Preview version)

[FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/MelPE) is a dedicated F0 predictor designed for real-time voice conversion and will become the preferred F0 predictor for sovits real-time voice conversion in the future.(The paper is being written)

If you are using the `fcpe` F0 Predictor, you will need to download the pre-trained FCPE model.

- download model at [fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)
  - Place it under the `pretrain` directory

## 📊 Dataset Preparation

Simply place the dataset in the `dataset_raw` directory with the following file structure:

```